#!/usr/bin/env python3
"""
Launch follower arm servers directly using the i2rt Python API.
No subprocess, no minimum_gello.py wrapper — full control over gripper limits.

Usage:
    # Use default open limit (GRIPPER_OPEN_LIMIT = 2.4)
    python start_follower_servers_direct.py

    # Override open limit from command line
    python start_follower_servers_direct.py --gripper_open 2.0

Gripper range: 0 ~ 2.7 (physical max). Change GRIPPER_OPEN_LIMIT below to adjust.
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
# Gripper open limit (range: 0 ~ 2.7, where 2.7 is fully open)
# Change this value to control how far the gripper can open.
# ============================================================
GRIPPER_OPEN_LIMIT = 2.4


def create_follower_server(can_channel: str, gripper_open_limit: float, port: int):
    """
    Create a follower robot and wrap it in a portal server.

    Uses get_yam_robot() with the gripper_open_limit parameter so autodetection
    runs first and then the open limit is overridden thread-safely.

    Args:
        can_channel: CAN interface name (e.g. "can_right", "can_left")
        gripper_open_limit: Motor position for the open limit (e.g. 2.4)
        port: Server port number
    """
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType

    gripper_type = GripperType.LINEAR_4310
    logger.info(f"[{can_channel}] Creating robot via get_yam_robot (gripper={gripper_type.value}, open_limit={gripper_open_limit})...")
    robot = get_yam_robot(
        channel=can_channel,
        gripper_type=gripper_type,
        gripper_open_limit=gripper_open_limit,
    )
    logger.info(f"[{can_channel}] Final gripper limits: {robot._gripper_limits}")

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
        description="Launch follower arm servers with custom gripper open limit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Default (auto-calibrate then cap open limit to GRIPPER_OPEN_LIMIT)
        python start_follower_servers_direct.py

        # Custom open limit
        python start_follower_servers_direct.py --gripper_open 2.0
        """,
    )

    parser.add_argument("--gripper_open", type=float, default=GRIPPER_OPEN_LIMIT,
                        help=f"Gripper open limit (default: {GRIPPER_OPEN_LIMIT}, full open: 2.7)")
    parser.add_argument("--right_port", type=int, default=1234,
                        help="Right follower server port (default: 1234)")
    parser.add_argument("--left_port", type=int, default=1235,
                        help="Left follower server port (default: 1235)")
    parser.add_argument("--right_can", type=str, default="can_right",
                        help="Right arm CAN channel (default: can_right)")
    parser.add_argument("--left_can", type=str, default="can_left",
                        help="Left arm CAN channel (default: can_left)")

    args = parser.parse_args()

    servers = []
    robots = []

    try:
        # Launch right follower
        print(f"Starting right follower on {args.right_can} port {args.right_port}...")
        right_robot, right_server = create_follower_server(
            args.right_can, args.gripper_open, args.right_port,
        )
        robots.append(right_robot)
        servers.append(right_server)
        print(f"  Right follower ready (gripper limits: {right_robot._gripper_limits})")

        # Launch left follower
        print(f"Starting left follower on {args.left_can} port {args.left_port}...")
        left_robot, left_server = create_follower_server(
            args.left_can, args.gripper_open, args.left_port,
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
