#!/usr/bin/env python
"""
Identify RealSense cameras by showing each one's live feed one at a time.
Press any key to move to the next camera.
"""

import cv2
import pyrealsense2 as rs

def get_connected_cameras():
    ctx = rs.context()
    devices = ctx.query_devices()
    cameras = []
    for dev in devices:
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        cameras.append((serial, name))
    return cameras

def show_camera(serial, index, total):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)
    
    pipeline.start(config)
    print(f"\n[Camera {index}/{total}] Serial: {serial}")
    print("  Look at the screen to identify this camera's position (left/right/top)")
    print("  Press any key to move to next camera, or 'q' to quit")
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            import numpy as np
            image = np.asanyarray(color_frame.get_data())
            
            # Add text overlay
            text = f"Camera {index}/{total} - Serial: {serial}"
            cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, "Press any key for next, 'q' to quit", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow(f"Identify Camera", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False
            elif key != 255:
                return True
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

def main():
    cameras = get_connected_cameras()
    print(f"Found {len(cameras)} RealSense cameras:")
    for i, (serial, name) in enumerate(cameras, 1):
        print(f"  {i}. {name} - Serial: {serial}")
    
    print("\nShowing each camera one by one...")
    print("Look at the live feed to determine which is left/right/top\n")
    
    results = {}
    for i, (serial, name) in enumerate(cameras, 1):
        cont = show_camera(serial, i, len(cameras))
        position = input(f"  Camera {serial} is which position? (left/right/top): ").strip().lower()
        results[position] = serial
        if not cont:
            break
    
    print("\n" + "="*60)
    print("CAMERA MAPPING:")
    print("="*60)
    for pos in ['left', 'right', 'top']:
        if pos in results:
            print(f"  {pos}: {results[pos]}")
    
    print("\nUpdate inference.sh with:")
    print(f"""  --robot.cameras='{{
    right: {{"type": "intelrealsense", "serial_number_or_name": "{results.get('right', '???')}", "width": 640, "height": 360, "fps": 30}},
    left:  {{"type": "intelrealsense", "serial_number_or_name": "{results.get('left', '???')}", "width": 640, "height": 360, "fps": 30}},
    top:   {{"type": "intelrealsense", "serial_number_or_name": "{results.get('top', '???')}", "width": 640, "height": 360, "fps": 30}}
  }}'""")

if __name__ == "__main__":
    main()
