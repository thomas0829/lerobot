# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example command:
```shell
python src/lerobot/async_inference/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
"""

import logging
import pickle  # nosec
import shutil
import sys
import termios
import threading
import time
import tty
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    bi_yam_follower,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR

from .configs import RobotClientConfig
from .constants import SUPPORTED_ROBOTS
from .helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)


def _wait_for_arrow_key():
    """Block until an arrow key or Ctrl+C is pressed.

    Returns:
        'right'  — → save and finish episode
        'left'   — ← cancel episode (discard, repeat)
        'ctrl_c' — Ctrl+C abort everything
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == "\x1b":          # ESC — start of arrow escape sequence
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    if ch3 == "C":   # right arrow: ESC [ C
                        return "right"
                    elif ch3 == "D": # left arrow:  ESC [ D
                        return "left"
            elif ch == "\x03":        # Ctrl+C
                return "ctrl_c"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """
        # Store configuration
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
        )
        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()

        # Initialize client side variables
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

        # Use an event for thread-safe coordination
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing

        # Per-episode done event (set to stop current episode loop)
        self.episode_done = threading.Event()

        # Recording setup
        self.dataset = None
        self._video_encoding_mgr = None
        self._frame_count = 0
        if config.record:
            self._setup_recording()

        # Home position (captured at startup or when explicitly set)
        self._home_position: dict[str, float] | None = None

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def _capture_home_position(self):
        """Read the robot's current joint positions and store as home."""
        obs = self.robot.get_observation()
        # Keep only keys that are action features (joint/gripper positions)
        self._home_position = {
            k: float(v) for k, v in obs.items() if k in self.robot.action_features
        }
        self.logger.info(f"Home position captured: {self._home_position}")

    def _return_home(self):
        """Smoothly interpolate from the current position back to home."""
        if self._home_position is None:
            self.logger.warning("No home position captured, skipping return_home.")
            return

        steps = self.config.return_home_steps
        dt = self.config.environment_dt

        # Read current position
        obs = self.robot.get_observation()
        current = {k: float(obs[k]) for k in self._home_position}

        self.logger.info(f"Returning to home over {steps} steps ({steps * dt:.1f}s)...")
        print(f"[Home] Returning to home position ({steps * dt:.1f}s)...")

        for step in range(1, steps + 1):
            alpha = step / steps
            target = {
                k: current[k] + alpha * (self._home_position[k] - current[k])
                for k in self._home_position
            }
            self.robot.send_action(target)
            time.sleep(dt)

        self.logger.info("Returned to home position.")
        print("[Home] Done.")

    def _setup_recording(self):
        """Initialize the LeRobot dataset for recording inference data."""
        # Build dataset features from robot's observation and action features
        obs_features = hw_to_dataset_features(
            self.robot.observation_features, OBS_STR, use_video=self.config.use_videos
        )
        action_features = hw_to_dataset_features(self.robot.action_features, ACTION)
        dataset_features = {**obs_features, **action_features}

        num_cameras = len([k for k in self.robot.observation_features if isinstance(self.robot.observation_features[k], tuple)])

        # Check if dataset directory already exists and ask user what to do
        dataset_root = (
            Path(self.config.dataset_root) if self.config.dataset_root else HF_LEROBOT_HOME / self.config.repo_id
        )
        if dataset_root.exists():
            print(f"\n[WARNING] Dataset directory already exists: {dataset_root}")
            print("  [d] Delete and start fresh")
            print("  [r] Rename (append timestamp suffix)")
            print("  [q] Quit")
            while True:
                choice = input("Choice [d/r/q]: ").strip().lower()
                if choice == "d":
                    shutil.rmtree(dataset_root)
                    print(f"[INFO] Deleted {dataset_root}")
                    break
                elif choice == "r":
                    import datetime
                    suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_path = dataset_root.parent / f"{dataset_root.name}_{suffix}"
                    dataset_root.rename(new_path)
                    print(f"[INFO] Renamed to {new_path}")
                    break
                elif choice == "q":
                    print("[INFO] Exiting.")
                    raise SystemExit(0)
                else:
                    print("  Please enter d, r, or q.")

        self.dataset = LeRobotDataset.create(
            repo_id=self.config.repo_id,
            fps=self.config.fps,
            root=self.config.dataset_root,
            robot_type=self.robot.name,
            features=dataset_features,
            use_videos=self.config.use_videos,
            image_writer_processes=0,
            image_writer_threads=num_cameras * 2,
        )

        # Enter VideoEncodingManager context for proper cleanup
        self._video_encoding_mgr = VideoEncodingManager(self.dataset)
        self._video_encoding_mgr.__enter__()

        self.logger.info(
            f"Recording enabled: repo_id={self.config.repo_id}, "
            f"features={list(dataset_features.keys())}, "
            f"use_videos={self.config.use_videos}"
        )

    def _record_frame(self, observation: dict, action: dict, task: str):
        """Record a single frame (observation + action) to the dataset."""
        if self.dataset is None:
            return

        try:
            obs_frame = build_dataset_frame(self.dataset.features, observation, prefix=OBS_STR)
            action_frame = build_dataset_frame(self.dataset.features, action, prefix=ACTION)
            frame = {**obs_frame, **action_frame, "task": task}
            self.dataset.add_frame(frame)
            self._frame_count += 1

            if self._frame_count % 100 == 0:
                self.logger.info(f"Recorded {self._frame_count} frames")
        except Exception as e:
            self.logger.error(f"Error recording frame: {e}")

    def _save_episode(self):
        """Save the current episode to the dataset (does NOT finalize/push)."""
        if self.dataset is None:
            return

        try:
            if self._frame_count > 0:
                self.logger.info(f"Saving episode with {self._frame_count} frames...")
                self.dataset.save_episode()
                self.logger.info(f"Episode saved. Total episodes: {self.dataset.meta.total_episodes}")
            else:
                self.logger.info("No frames recorded, skipping episode save.")
        except Exception as e:
            self.logger.error(f"Error saving episode: {e}")

    def _finalize_recording(self):
        """Finalize and close the dataset (call once after all episodes)."""
        if self.dataset is None:
            return

        try:
            # Exit VideoEncodingManager context (triggers finalize + video encoding)
            if self._video_encoding_mgr is not None:
                self._video_encoding_mgr.__exit__(None, None, None)
                self._video_encoding_mgr = None

            self.logger.info(f"Dataset finalized at {self.dataset.root}")

            # Upload to HuggingFace Hub
            if self.config.push_to_hub:
                self.logger.info(f"Uploading dataset to HuggingFace Hub: {self.dataset.repo_id} ...")
                print(f"[INFO] Uploading to HuggingFace Hub ({self.dataset.repo_id})...")
                self.dataset.push_to_hub()
                self.logger.info(f"Dataset uploaded: https://huggingface.co/datasets/{self.dataset.repo_id}")
                print(f"[INFO] Upload complete: https://huggingface.co/datasets/{self.dataset.repo_id}")
        except Exception as e:
            self.logger.error(f"Error finalizing recording: {e}")
        finally:
            # Clear dataset so any accidental second call is a no-op
            self.dataset = None

    def _save_recording(self):
        """Save the current episode and finalize the dataset (single-episode convenience wrapper)."""
        self._save_episode()
        self._finalize_recording()

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client"""
        self.shutdown_event.set()

        # Finalize recording before disconnecting (episodes were already saved)
        self._finalize_recording()

        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.debug(f"Sent observation #{obs_timestep} | ")

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= latest_action:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # If the new action's timestep is in the current action queue, aggregate it
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                # Use StreamActions to get a stream of actions from the server
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Log incoming actions
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                # Update action queue
                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!

                if verbose:
                    # Get queue state after changes
                    new_size, new_timesteps = self._inspect_action_queue()

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self):
        """Check if there are actions available in the queue"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """Reading and performing actions in local queue"""

        # Lock only for queue operations
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            # Get action from queue
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        _performed_action = self.robot.send_action(
            self._action_tensor_to_action_dict(timed_action.get_action())
        )
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return _performed_action

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            raw_observation: RawObservation = self.robot.get_observation()
            raw_observation["task"] = task

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            _ = self.send_observation(observation)

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

    def control_loop(
        self, task: str, verbose: bool = False, episode_timeout_s: float = 0.0
    ) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations.

        Runs until shutdown, episode_done is set, or episode_timeout_s is exceeded.
        """
        _performed_action = None
        _last_recorded_action = None  # last action used for recording (reused when queue is empty)
        raw_observation = None

        episode_start = time.perf_counter()

        while self.running and not self.episode_done.is_set():
            control_loop_start = time.perf_counter()

            # Check episode timeout
            if episode_timeout_s > 0 and (time.perf_counter() - episode_start) >= episode_timeout_s:
                self.logger.info(f"Episode timeout reached ({episode_timeout_s:.1f}s), ending episode.")
                self.episode_done.set()
                break

            """Control loop: (1) Always capture a fresh observation from the robot (for recording + sending)"""
            raw_observation: RawObservation = self.robot.get_observation()
            raw_observation["task"] = task

            # Extract actual current joint positions from observation (used as fallback action for recording)
            current_joint_action = {k: v for k, v in raw_observation.items() if k in self.robot.action_features}

            """Control loop: (2) Performing actions, when available"""
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)
                _last_recorded_action = _performed_action
            else:
                # When queue is empty, record actual joint state instead of repeating stale command
                _last_recorded_action = current_joint_action

            """Control loop: (3) Send observation to server when queue is low enough"""
            if self._ready_to_send_observation():
                with self.latest_action_lock:
                    latest_action = self.latest_action
                observation = TimedObservation(
                    timestamp=time.time(),
                    observation=raw_observation,
                    timestep=max(latest_action, 0),
                )
                with self.action_queue_lock:
                    observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                _ = self.send_observation(observation)
                if observation.must_go:
                    self.must_go.clear()
                if verbose:
                    fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())
                    self.logger.info(
                        f"Obs #{observation.get_timestep()} | "
                        f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                        f"Target: {fps_metrics['target_fps']:.2f}"
                    )

            """Control loop: (4) Record frame every loop at full fps.
            Use the last known action when no new action was executed this step."""
            if self.config.record and _last_recorded_action is not None:
                self._record_frame(raw_observation, _last_recorded_action, task)

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return raw_observation, _performed_action

    def run_episodes(self, task: str, num_episodes: int, episode_timeout_s: float = 0.0, verbose: bool = False):
        """Run multiple episodes, prompting for Enter before each one.

        Press Enter to start each episode.
        While the episode is running, press Enter again to stop it early.
        The episode also ends automatically when episode_timeout_s is exceeded (if > 0).
        After all episodes the dataset is finalized.
        """
        # Sync with the receive_actions thread (barrier)
        self.start_barrier.wait()
        self.logger.info("Episode runner starting")

        # Capture home position from the robot's current state
        if self.config.return_home:
            self._capture_home_position()

        timeout_str = f"{episode_timeout_s:.0f}s timeout" if episode_timeout_s > 0 else "no timeout"

        ep_idx = 0
        while ep_idx < num_episodes:
            print(f"\n[Episode {ep_idx + 1}/{num_episodes}] Press Enter to start ({timeout_str})...")
            try:
                input()
            except KeyboardInterrupt:
                print("\nAborted by user.")
                break

            # Reset per-episode state
            self.episode_done.clear()
            with self.action_queue_lock:
                self.action_queue = Queue()
            with self.latest_action_lock:
                self.latest_action = -1
            self.action_chunk_size = -1
            self.must_go.set()
            self._frame_count = 0

            print(f"[Episode {ep_idx + 1}/{num_episodes}] Running... (→ save & next | ← cancel & retry)")

            # Run control loop in a background thread so the main thread can
            # listen for arrow keys to control the episode.
            control_thread = threading.Thread(
                target=self.control_loop,
                kwargs={"task": task, "verbose": verbose, "episode_timeout_s": episode_timeout_s},
                daemon=True,
            )
            control_thread.start()

            # Block until an arrow key or Ctrl+C.
            key = _wait_for_arrow_key()
            self.episode_done.set()
            control_thread.join()

            if key == "ctrl_c":
                print(f"\n[Episode {ep_idx + 1}/{num_episodes}] Aborted.")
                break

            # Return to home regardless of save/cancel
            if self.config.return_home:
                self._return_home()

            if key == "left":
                print(f"[Episode {ep_idx + 1}/{num_episodes}] Cancelled — retrying same episode.")
                # Do NOT save and do NOT increment ep_idx
            else:  # 'right' or timeout
                print(f"[Episode {ep_idx + 1}/{num_episodes}] Saving episode...")
                self._save_episode()
                ep_idx += 1

        # Finalize dataset after all episodes
        self._finalize_recording()


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")

        # Create and start action receiver thread
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()

        try:
            # The main thread runs the episode loop (handles barrier sync internally)
            client.run_episodes(
                task=cfg.task,
                num_episodes=cfg.num_episodes,
                episode_timeout_s=cfg.episode_timeout_s,
            )

        except KeyboardInterrupt:
            client.logger.info("Interrupted by user.")

        finally:
            client.stop()
            action_receiver_thread.join(timeout=5)
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    async_client()  # run the client
