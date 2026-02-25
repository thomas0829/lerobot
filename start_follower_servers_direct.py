#!/usr/bin/env python3
"""
Launch follower arm servers directly using the i2rt Python API.
No subprocess, no minimum_gello.py wrapper — full control over gripper limits.

Usage:
    # Default: override gripper limits to [0.0, -2.4] (matching data collection)
    python start_follower_servers_direct.py

    # Custom limits
    python start_follower_servers_direct.py --gripper_open 0.0 --gripper_close -2.4

    # Use auto-calibrated values (no override)
    python start_follower_servers_direct.py --no_override

The auto-calibration detects physical close/open endpoints, but these may differ
from the limits used during data collection. This script overrides the JointMapper
so that cmd=0 and cmd=1 map to the same motor positions as during data collection.
"""

import argparse
import logging
import signal
import sys
import threading
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Gripper limits override.
# Set both open and close motor positions to match data collection.
# Set to None to use auto-calibrated values.
# ============================================================
GRIPPER_CLOSE_POS = 0.0    # motor position when gripper is fully closed (cmd=0)
GRIPPER_OPEN_POS = -2.4    # motor position when gripper is fully open (cmd=1)


def create_follower_server(can_channel: str, port: int,
                           gripper_open_pos: float = None,
                           gripper_close_pos: float = None):
    """
    Create a follower robot and wrap it in a portal server.

    Calls get_yam_robot() for auto-calibration, then overrides the gripper
    limits to match the values used during data collection.

    The JointMapper maps (matching auto-cal convention):
        cmd=0 -> limits[0] (close end)
        cmd=1 -> limits[1] (open end)

    Args:
        can_channel: CAN interface name (e.g. "can_right", "can_left")
        port: Server port number
        gripper_open_pos: Motor position for fully open (cmd=1). None = use auto-cal.
        gripper_close_pos: Motor position for fully closed (cmd=0). None = use auto-cal.
    """
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType, JointMapper

    gripper_type = GripperType.LINEAR_4310
    logger.info(f"[{can_channel}] Creating robot via get_yam_robot (gripper={gripper_type.value})...")
    robot = get_yam_robot(
        channel=can_channel,
        gripper_type=gripper_type,
    )
    auto_limits = np.array(robot._gripper_limits, dtype=float)
    logger.info(f"[{can_channel}] Auto-calibrated gripper limits: {auto_limits}")
    logger.info(f"[{can_channel}] Auto-cal: limits[0](close)={auto_limits[0]:.4f}, "
                f"limits[1](open)={auto_limits[1]:.4f}")

    # Override gripper limits if specified
    # Convention: limits[0] = close position (cmd=0), limits[1] = open position (cmd=1)
    new_limits = auto_limits.copy()
    if gripper_close_pos is not None:
        new_limits[0] = gripper_close_pos
    if gripper_open_pos is not None:
        new_limits[1] = gripper_open_pos

    if gripper_open_pos is not None or gripper_close_pos is not None:
        # Build the new JointMapper
        new_remapper = JointMapper(
            index_range_map={robot._gripper_index: new_limits},
            total_dofs=len(robot.motor_chain),
        )

        # Thread-safe swap
        with robot._command_lock:
            robot._gripper_limits = new_limits
            robot.remapper = new_remapper

        logger.info(f"[{can_channel}] Gripper limits overridden: {auto_limits} -> {new_limits}")

        # After override, command the gripper to the open position (cmd=1)
        # so it moves back within the new range instead of staying at the
        # auto-calibrated physical max.
        current_pos = robot.get_joint_pos()
        current_pos[robot._gripper_index] = 1.0  # cmd=1 = fully open in new range
        robot.command_joint_pos(current_pos)
        logger.info(f"[{can_channel}] Commanded gripper to open position (cmd=1, motor={new_limits[1]:.4f})")
        time.sleep(1.0)  # give gripper time to move

    # Verify: check what cmd=0 and cmd=1 map to in motor space
    logger.info(f"[{can_channel}] Final gripper limits: {robot._gripper_limits}")
    test_cmd = np.zeros(len(robot.motor_chain))
    test_cmd[robot._gripper_index] = 0.0
    motor_at_0 = robot.remapper.to_robot_joint_pos_space(test_cmd)[robot._gripper_index]
    test_cmd[robot._gripper_index] = 1.0
    motor_at_1 = robot.remapper.to_robot_joint_pos_space(test_cmd)[robot._gripper_index]
    logger.info(f"[{can_channel}] Verification: cmd=0 -> motor={motor_at_0:.4f}, "
                f"cmd=1 -> motor={motor_at_1:.4f}, range={abs(motor_at_1 - motor_at_0):.4f}")

    # Create portal server (same as ServerRobot in minimum_gello.py)
    import portal
    server = portal.Server(port)
    server.bind("num_dofs", robot.num_dofs)
    server.bind("get_joint_pos", robot.get_joint_pos)
    server.bind("command_joint_pos", robot.command_joint_pos)
    server.bind("command_joint_state", robot.command_joint_state)
    server.bind("get_observations", robot.get_observations)

    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()
    time.sleep(0.5)  # Give the server a moment to bind

    return robot, server


def main():
    parser = argparse.ArgumentParser(
        description="Launch follower arm servers with custom gripper limits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: use limits from constants (open=0.0, close=-2.4)
  python start_follower_servers_direct.py

  # Custom limits
  python start_follower_servers_direct.py --gripper_open -2.4 --gripper_close 0.0

  # Use auto-calibrated values (no override)
  python start_follower_servers_direct.py --no_override
        """,
    )

    parser.add_argument("--gripper_open", type=float, default=GRIPPER_OPEN_POS,
                        help=f"Motor position for fully open gripper, cmd=1 "
                             f"(default: {GRIPPER_OPEN_POS})")
    parser.add_argument("--gripper_close", type=float, default=GRIPPER_CLOSE_POS,
                        help=f"Motor position for fully closed gripper, cmd=0 "
                             f"(default: {GRIPPER_CLOSE_POS})")
    parser.add_argument("--no_override", action="store_true",
                        help="Use auto-calibrated values, do not override")
    parser.add_argument("--right_port", type=int, default=1234,
                        help="Right follower server port (default: 1234)")
    parser.add_argument("--left_port", type=int, default=1235,
                        help="Left follower server port (default: 1235)")
    parser.add_argument("--right_can", type=str, default="can_right",
                        help="Right arm CAN channel (default: can_right)")
    parser.add_argument("--left_can", type=str, default="can_left",
                        help="Left arm CAN channel (default: can_left)")

    args = parser.parse_args()

    gripper_open = None if args.no_override else args.gripper_open
    gripper_close = None if args.no_override else args.gripper_close

    servers = []
    robots = []

    try:
        # Launch right follower
        print(f"Starting right follower on {args.right_can} port {args.right_port}...")
        right_robot, right_server = create_follower_server(
            args.right_can, args.right_port,
            gripper_open_pos=gripper_open,
            gripper_close_pos=gripper_close,
        )
        robots.append(right_robot)
        servers.append(right_server)
        print(f"  Right follower ready (gripper limits: {right_robot._gripper_limits})")

        # Launch left follower
        print(f"Starting left follower on {args.left_can} port {args.left_port}...")
        left_robot, left_server = create_follower_server(
            args.left_can, args.left_port,
            gripper_open_pos=gripper_open,
            gripper_close_pos=gripper_close,
        )
        robots.append(left_robot)
        servers.append(left_server)
        print(f"  Left follower ready (gripper limits: {left_robot._gripper_limits})")

        print("\n=== Both follower servers ready for inference ===")
        print(f"  Right: localhost:{args.right_port}")
        print(f"  Left:  localhost:{args.left_port}")
        print("Press Ctrl+C to stop\n")

        # Keep alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down servers...")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
