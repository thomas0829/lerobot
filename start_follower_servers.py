#!/usr/bin/env python3
"""
Launch only the follower arm servers for inference (no leader arms needed).
Supports custom gripper limits.
"""

import argparse
import os
import signal
import subprocess
import sys
import time


def find_i2rt_script():
    """Find the i2rt minimum_gello.py script."""
    try:
        import i2rt
        i2rt_path = os.path.dirname(i2rt.__file__)
        script_path = os.path.join(os.path.dirname(i2rt_path), "scripts", "minimum_gello.py")
        if os.path.exists(script_path):
            return script_path
    except ImportError:
        raise RuntimeError("Could not import i2rt. Please install it.")
    raise RuntimeError("Could not find i2rt minimum_gello.py script.")


def launch_server(can_channel, gripper, mode, server_port, gripper_open_limit=None, gripper_close_limit=None):
    """Launch a single server process."""
    script_path = find_i2rt_script()
    
    cmd = [
        sys.executable,
        script_path,
        "--can_channel", can_channel,
        "--gripper", gripper,
        "--mode", mode,
        "--server_port", str(server_port),
    ]
    
    # Add gripper limits if specified
    if gripper_open_limit is not None:
        cmd.extend(["--gripper_open_limit", str(gripper_open_limit)])
    if gripper_close_limit is not None:
        cmd.extend(["--gripper_close_limit", str(gripper_close_limit)])
    
    print(f"Starting: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def main():
    parser = argparse.ArgumentParser(description="Launch follower arm servers for inference")
    parser.add_argument("--gripper_open", type=float, default=None,
                        help="Gripper open limit (max position). If not set, uses auto-calibration.")
    parser.add_argument("--gripper_close", type=float, default=None,
                        help="Gripper close limit (min position). If not set, uses auto-calibration.")
    args = parser.parse_args()
    
    processes = []
    
    try:
        # Launch only follower servers
        print("Launching follower arm servers...")
        if args.gripper_open is not None or args.gripper_close is not None:
            print(f"Using custom gripper limits: open={args.gripper_open}, close={args.gripper_close}")
        
        # Right follower on port 1234
        p1 = launch_server("can_right", "linear_4310", "follower", 1234,
                          gripper_open_limit=args.gripper_open,
                          gripper_close_limit=args.gripper_close)
        processes.append(p1)
        print(f"✓ Right follower: localhost:1234 (PID {p1.pid})")
        
        # Left follower on port 1235
        p2 = launch_server("can_left", "linear_4310", "follower", 1235,
                          gripper_open_limit=args.gripper_open,
                          gripper_close_limit=args.gripper_close)
        processes.append(p2)
        print(f"✓ Left follower: localhost:1235 (PID {p2.pid})")
        
        print("\n✓ Follower servers ready for inference")
        print("Press Ctrl+C to stop servers\n")
        
        # Keep running
        while True:
            for process in processes:
                if process.poll() is not None:
                    print(f"Process {process.pid} terminated")
                    return
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping servers...")
    finally:
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
        print("Servers stopped")


if __name__ == "__main__":
    main()
