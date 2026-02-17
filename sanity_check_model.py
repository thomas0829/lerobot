#!/usr/bin/env python
"""
Sanity check: Load real images from dataset, feed to model, compare predicted vs ground truth actions.
No robot needed - pure offline test.

Usage:
    python sanity_check_model.py --model sengi/pi05_put_dolls_cloth --dataset thomas0829/put_the_dolls_on_the_cloth
    python sanity_check_model.py --model Jiafei1224/b --dataset Jiafei1224/a
"""

import argparse
import torch
import numpy as np
import json
from pathlib import Path
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import snapshot_download


def find_local_cache_path(model_id):
    """Find the local HF cache path for a model."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_dir / f"models--{model_id.replace('/', '--')}"
    if model_dir.exists():
        snapshots = model_dir / "snapshots"
        if snapshots.exists():
            # Get the latest snapshot
            subdirs = sorted(snapshots.iterdir())
            if subdirs:
                return str(subdirs[-1])
    return None


def clean_config(model_path):
    """Download complete model snapshot and check/clean config."""
    # Download complete snapshot (all files) if model is from Hub
    if "/" in model_path:
        print(f"Downloading complete model snapshot from Hub...")
        local_path = snapshot_download(repo_id=model_path, repo_type="model")
        config_path = Path(local_path) / "config.json"
    else:
        local_path = model_path
        config_path = Path(model_path) / "config.json"
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Invalid fields for PI05Config
    invalid_fields = [
        'compiled', 'attention_implementation', 'use_lora', 
        'lora_rank', 'lora_alpha', 'lora_dropout', 'lora_target_modules'
    ]
    
    found_invalid = {k: v for k, v in config.items() if k in invalid_fields}
    
    if found_invalid:
        print("\n[INFO] Detected training-specific config fields:")
        for k, v in found_invalid.items():
            print(f"  - {k}: {v}")
        
        # Check if LoRA was used
        if 'use_lora' in found_invalid and found_invalid['use_lora']:
            print("\n  [LoRA Analysis]")
            print(f"    Rank: {found_invalid.get('lora_rank', 'N/A')}")
            print(f"    Alpha: {found_invalid.get('lora_alpha', 'N/A')}")
            print(f"    Target modules: {found_invalid.get('lora_target_modules', 'N/A')}")
            print("    Note: LoRA adapters should already be merged into model weights")
        
        if 'compiled' in found_invalid and found_invalid['compiled']:
            print("\n  [Model Compilation]")
            print("    Model was compiled with torch.compile() during training")
        
        # These fields are not valid for PI05Config and must be removed to load
        print("\n  [Action] Removing invalid fields from local cached config.json")
        print(f"    Path: {config_path}")
        print("    Note: This only modifies local cache, not the Hub repository")
        
        # Remove invalid fields
        modified = False
        for k in invalid_fields:
            if k in config:
                del config[k]
                modified = True
        
        if modified:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print("    [OK] Config cleaned\n")
    
    return local_path


def load_model_and_processors(model_path, device="cuda"):
    print(f"Loading model: {model_path}")
    
    # Download complete snapshot and clean config
    local_path = clean_config(model_path)
    
    # Load model from local snapshot
    policy = PI05Policy.from_pretrained(local_path)
    policy.eval()
    policy.to(device)

    # Load preprocessor/postprocessor from same local path
    print(f"Loading processors from: {local_path}")

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        local_path, config_filename="policy_preprocessor.json"
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        local_path, config_filename="policy_postprocessor.json"
    )
    print(f"[OK] Model loaded on {device}")
    return policy, preprocessor, postprocessor


def print_processor_stats(postprocessor):
    """Print the unnormalization stats from the postprocessor."""
    print("\n" + "=" * 80)
    print("Postprocessor (unnormalization) stats:")
    print("=" * 80)
    for step in postprocessor.steps:
        sd = step.state_dict() if hasattr(step, 'state_dict') else {}
        if "action.min" in sd:
            action_min = sd["action.min"].cpu().numpy()
            action_max = sd["action.max"].cpu().numpy()
            print(f"  action.min: {action_min}")
            print(f"  action.max: {action_max}")
        if "action.mean" in sd:
            action_mean = sd["action.mean"].cpu().numpy()
            action_std = sd["action.std"].cpu().numpy()
            print(f"  action.mean: {action_mean}")
            print(f"  action.std:  {action_std}")


def dataset_item_to_model_input(item, device="cuda"):
    """Convert a dataset item to model input format.
    Let the preprocessor handle batching - just pass raw tensors."""
    obs = {}

    # Key mapping: dataset keys -> model expected keys
    key_remap = {
        "observation.images.left_camera": "observation.images.left",
        "observation.images.front_camera": "observation.images.top",
        "observation.images.right_camera": "observation.images.right",
    }

    # Find state and image keys - pass as-is, preprocessor will batch
    for key in item:
        mapped_key = key_remap.get(key, key)
        if key.startswith("observation.state"):
            obs[mapped_key] = item[key]
        elif key.startswith("observation.images."):
            obs[mapped_key] = item[key]

    # Task
    if "task" in item:
        obs["task"] = item["task"]
    else:
        obs["task"] = "Put the dolls on the cloth."

    return obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="sengi/pi05_put_dolls_cloth")
    parser.add_argument("--dataset", type=str, default="thomas0829/put_the_dolls_on_the_cloth")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2],
                        help="Episodes to test")
    parser.add_argument("--samples_per_episode", type=int, default=5,
                        help="Number of samples to test per episode")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Load model
    policy, preprocessor, postprocessor = load_model_and_processors(args.model, args.device)
    print_processor_stats(postprocessor)

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    dataset = LeRobotDataset(repo_id=args.dataset, episodes=args.episodes, video_backend="pyav")
    print(f"[OK] Dataset loaded: {len(dataset)} frames")

    # Show dataset keys
    sample = dataset[0]
    print(f"\nDataset keys: {list(sample.keys())}")

    # Show dataset stats
    if hasattr(dataset, 'meta') and hasattr(dataset.meta, 'stats'):
        stats = dataset.meta.stats
        if "action" in stats:
            print(f"\nDataset action stats:")
            for stat_key in stats["action"]:
                val = stats["action"][stat_key]
                arr = val.numpy() if hasattr(val, 'numpy') else np.array(val)
                print(f"  {stat_key}: {arr}")

    # Get action feature names
    action_names = []
    for key in sample:
        if key == "action":
            action_dim = sample["action"].shape[-1]
            # Try to infer names from dataset
            action_names = [f"action[{i}]" for i in range(action_dim)]
            break

    # Test on real data
    print("\n" + "=" * 80)
    print("TESTING MODEL ON REAL DATASET IMAGES")
    print("=" * 80)

    all_errors = []
    all_gt_actions = []
    all_pred_actions = []

    total_frames = len(dataset)

    for ep in args.episodes:
        # Find frames for this episode
        ep_indices = [i for i in range(total_frames) if dataset[i].get("episode_index", -1) == ep]
        if not ep_indices:
            print(f"\nEpisode {ep}: no frames found, skipping")
            continue

        # Sample evenly from episode
        step = max(1, len(ep_indices) // args.samples_per_episode)
        sample_indices = ep_indices[::step][:args.samples_per_episode]

        print(f"\nEpisode {ep}: {len(ep_indices)} frames, testing {len(sample_indices)} samples")

        for idx in sample_indices:
            item = dataset[idx]
            gt_action = item["action"].numpy()
            frame_idx = item.get("frame_index", idx)

            # Convert to model input
            obs = dataset_item_to_model_input(item, args.device)

            # Run inference
            with torch.no_grad():
                policy.reset()
                processed_obs = preprocessor(obs)
                output = policy.select_action(processed_obs)
                # select_action may return a tensor or dict
                if isinstance(output, torch.Tensor):
                    output_dict = {"action": output}
                else:
                    output_dict = output
                
                # Get normalized prediction (before postprocessing)
                norm_pred = output_dict["action"].cpu().numpy().flatten()
                
                # Apply postprocessing (unnormalize)
                final_output = postprocessor(output_dict)
                pred_action = final_output["action"].cpu().numpy().flatten()

            # Compare
            error = np.abs(pred_action - gt_action)
            all_errors.append(error)
            all_gt_actions.append(gt_action)
            all_pred_actions.append(pred_action)

            print(f"\n  Frame {frame_idx}:")
            print(f"    {'Dim':<12} {'GT':>10} {'Pred':>10} {'Norm':>10} {'Error':>10}")
            print(f"    {'-'*56}")
            for i in range(len(gt_action)):
                marker = " ← GRIPPER" if i == 6 or i == 13 else ""
                print(f"    {action_names[i]:<12} {gt_action[i]:>10.4f} {pred_action[i]:>10.4f} {norm_pred[i]:>10.4f} {error[i]:>10.4f}{marker}")

    if not all_errors:
        print("\nNo samples tested!")
        return

    # Summary
    all_errors = np.array(all_errors)
    all_gt_actions = np.array(all_gt_actions)
    all_pred_actions = np.array(all_pred_actions)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nTotal samples tested: {len(all_errors)}")
    print(f"\nPer-dimension Mean Absolute Error (MAE):")
    print(f"  {'Dim':<12} {'MAE':>10} {'GT range':>15} {'Pred range':>15}")
    print(f"  {'-'*54}")
    for i in range(all_errors.shape[1]):
        gt_range = f"[{all_gt_actions[:, i].min():.3f}, {all_gt_actions[:, i].max():.3f}]"
        pred_range = f"[{all_pred_actions[:, i].min():.3f}, {all_pred_actions[:, i].max():.3f}]"
        marker = " ← GRIPPER" if i == 6 or i == 13 else ""
        print(f"  {action_names[i]:<12} {all_errors[:, i].mean():>10.4f} {gt_range:>15} {pred_range:>15}{marker}")

    overall_mae = all_errors.mean()
    print(f"\nOverall MAE: {overall_mae:.4f}")

    # Diagnosis
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    if overall_mae > 0.5:
        print("[FAIL] MAE > 0.5: Model predictions are far from ground truth.")
        print("   Possible causes:")
        print("   - Model not trained enough")
        print("   - Preprocessor/postprocessor normalization mismatch")
        print("   - Wrong model checkpoint")
    elif overall_mae > 0.2:
        print("[WARN] MAE 0.2~0.5: Model predictions are somewhat off.")
        print("   May work on robot but with reduced accuracy.")
    elif overall_mae > 0.05:
        print("[OK] MAE 0.05~0.2: Reasonable for a fine-tuned model.")
        print("   Should be testable on robot.")
    else:
        print("[PASS] MAE < 0.05: Model predictions closely match ground truth.")
        print("   Looks good for deployment!")

    # Check for NaN or inf
    if np.any(np.isnan(all_pred_actions)) or np.any(np.isinf(all_pred_actions)):
        print("\n[ERROR] WARNING: NaN or Inf detected in predictions!")

    # Check gripper values
    for grip_idx in [6, 13]:
        if grip_idx < all_pred_actions.shape[1]:
            grip_preds = all_pred_actions[:, grip_idx]
            grip_gts = all_gt_actions[:, grip_idx]
            print(f"\n  Gripper [{grip_idx}]: pred range [{grip_preds.min():.3f}, {grip_preds.max():.3f}], "
                  f"GT range [{grip_gts.min():.3f}, {grip_gts.max():.3f}]")
            if grip_preds.max() > 1.5 or grip_preds.min() < -0.5:
                print(f"  [ERROR] Gripper [{grip_idx}] predictions out of expected [0, 1] range!")


if __name__ == "__main__":
    main()
