#!/bin/bash

python -m lerobot.async_inference.robot_client \
  --server_address=localhost:8000 \
  --robot.type=bi_yam_follower \
  --robot.left_arm_port=1235 \
  --robot.right_arm_port=1234 \
  --robot.cameras='{
    right: {"type": "intelrealsense", "serial_number_or_name": "128422272697", "width": 640, "height": 360, "fps": 30},
    left:  {"type": "intelrealsense", "serial_number_or_name": "218622275075", "width": 640, "height": 360, "fps": 30},
    top:   {"type": "intelrealsense", "serial_number_or_name": "215222073684", "width": 640, "height": 360, "fps": 30}
  }' \
  --task="Put the dolls on the cloth." \
  --policy_type=pi05 \
  --pretrained_name_or_path=sengi/pi05_put_dolls_cloth_lerobot \
  --policy_device=cuda \
  --actions_per_chunk=30 \
  --chunk_size_threshold=0.0 \
  --aggregate_fn_name=weighted_average \
  --record=true \
  --repo_id=thomas0829/inference_put_dolls_cloth \
  --use_videos=true


#python start_follower_servers.py
# python -m lerobot.async_inference.policy_server \
#   --host=0.0.0.0 \
#   --port=8000 \
#   --fps=30