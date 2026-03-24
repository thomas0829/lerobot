#!/usr/bin/env python3
"""
Launch follower arm servers using i2rt Python API with gripper limit remapping.

After auto-calibration, the JointMapper is rebuilt so that cmd=0..1 maps to
the desired motor-position range (default: 0.0 to -5.2825).

The gripper stays in zero-gravity mode (kp=0, kd=0) at startup — exactly
like minimum_gello.py follower mode. No active gripper command is sent until
the inference client calls command_joint_pos().

Usage:
    # Default: auto-cal then remap gripper to [0.0, -5.2825]
    python start_follower_servers_v2.py

    # Custom limits
    python start_follower_servers_v2.py --gripper_open -4.0 --gripper_close 0.0

    # Use auto-calibrated values only (no remap)
    python start_follower_servers_v2.py --no_override
"""

import argparse
import logging
import sys
import threading
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Gripper limits (motor-position space).
#   cmd=0  →  GRIPPER_CLOSE_POS  (close end)
#   cmd=1  →  GRIPPER_OPEN_POS   (open end)
# ============================================================
GRIPPER_CLOSE_POS = 0.0
GRIPPER_OPEN_POS = -5.2


def create_follower_server(
    can_channel: str,
    port: int,
    gripper_open_pos: float = None,
    gripper_close_pos: float = None,
):
    """
    Create a follower robot and wrap it in a portal server.

    Steps:
      1. get_yam_robot() auto-calibrates the gripper (torque probing).
      2. If override limits are given, rebuild the JointMapper so cmd 0..1
         maps to [close_pos, open_pos] instead of auto-cal values.
      3. Do NOT send any gripper command — stays in zero-gravity mode.
    """
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType, JointMapper

    gripper_type = GripperType.LINEAR_4310
    logger.info(f"[{can_channel}] Creating robot (gripper={gripper_type.value})…")
    robot = get_yam_robot(channel=can_channel, gripper_type=gripper_type)

    auto_limits = np.array(robot._gripper_limits, dtype=float)
    logger.info(
        f"[{can_channel}] Auto-cal gripper limits: "
        f"close={auto_limits[0]:.4f}, open={auto_limits[1]:.4f}"
    )

    # ---- Override JointMapper if requested ----
    need_override = gripper_open_pos is not None or gripper_close_pos is not None
    if need_override:
        new_limits = auto_limits.copy()
        if gripper_close_pos is not None:
            new_limits[0] = gripper_close_pos
        if gripper_open_pos is not None:
            new_limits[1] = gripper_open_pos

        new_remapper = JointMapper(
            index_range_map={robot._gripper_index: new_limits},
            total_dofs=len(robot.motor_chain),
        )

        # Atomic swap under both locks (update() reads remapper under _state_lock)
        with robot._command_lock:
            with robot._state_lock:
                robot._gripper_limits = new_limits
                robot.remapper = new_remapper

        logger.info(f"[{can_channel}] Gripper limits overridden: {auto_limits} → {new_limits}")

    # ---- Verify mapping ----
    limits = robot._gripper_limits
    test = np.zeros(len(robot.motor_chain))
    test[robot._gripper_index] = 0.0
    m0 = robot.remapper.to_robot_joint_pos_space(test)[robot._gripper_index]
    test[robot._gripper_index] = 1.0
    m1 = robot.remapper.to_robot_joint_pos_space(test)[robot._gripper_index]
    logger.info(
        f"[{can_channel}] Final mapping: cmd=0→motor {m0:.4f}, "
        f"cmd=1→motor {m1:.4f}, range={abs(m1 - m0):.4f}"
    )

    # ---- Portal server (same bindings as minimum_gello.py ServerRobot) ----
    import portal

    server = portal.Server(port)
    server.bind("num_dofs", robot.num_dofs)
    server.bind("get_joint_pos", robot.get_joint_pos)
    server.bind("command_joint_pos", robot.command_joint_pos)
    server.bind("command_joint_state", robot.command_joint_state)
    server.bind("get_observations", robot.get_observations)

    threading.Thread(target=server.start, daemon=True).start()
    time.sleep(0.5)

    return robot, server


def main():
    parser = argparse.ArgumentParser(
        description="Launch follower arm servers with gripper limit remapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_follower_servers_v2.py
  python start_follower_servers_v2.py --gripper_open -4.0 --gripper_close 0.0
  python start_follower_servers_v2.py --no_override
""",
    )
    parser.add_argument(
        "--gripper_open", type=float, default=GRIPPER_OPEN_POS,
        help=f"Motor pos for fully open (cmd=1). Default: {GRIPPER_OPEN_POS}",
    )
    parser.add_argument(
        "--gripper_close", type=float, default=GRIPPER_CLOSE_POS,
        help=f"Motor pos for fully closed (cmd=0). Default: {GRIPPER_CLOSE_POS}",
    )
    parser.add_argument(
        "--no_override", action="store_true",
        help="Use auto-calibrated values, do not remap",
    )
    parser.add_argument("--right_port", type=int, default=1234)
    parser.add_argument("--left_port", type=int, default=1235)
    parser.add_argument("--right_can", type=str, default="can_right")
    parser.add_argument("--left_can", type=str, default="can_left")
    args = parser.parse_args()

    g_open = None if args.no_override else args.gripper_open
    g_close = None if args.no_override else args.gripper_close

    robots = []

    try:
        for label, can, port in [
            ("Right", args.right_can, args.right_port),
            ("Left", args.left_can, args.left_port),
        ]:
            print(f"Starting {label} follower on {can} port {port}…")
            robot, _ = create_follower_server(can, port, g_open, g_close)
            robots.append(robot)
            print(f"  {label} ready  (gripper limits: {robot._gripper_limits})")

        print(f"\n=== Both followers ready ===")
        print(f"  Right: localhost:{args.right_port}")
        print(f"  Left:  localhost:{args.left_port}")
        print("Press Ctrl+C to stop\n")

        while True:
            for robot in robots:
                if not robot.motor_chain.running:
                    logger.error(f"Motor chain stopped: {robot}")
                if not robot._server_thread.is_alive():
                    logger.error(f"Server thread died: {robot}")
            time.sleep(2)

    except KeyboardInterrupt:
        print("\nShutting down…")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
