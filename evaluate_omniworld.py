"""
Configuration for OmniWorld-Game → DPVO evaluation.

Known data format:
- color/000000.png ... (6-digit zero-padded)
- camera/split_0.json ... split_N.json
  - focals: [float, ...] per-frame (fx = fy)
  - quats: [[w,x,y,z], ...] unnormalized
  - trans: [[tx,ty,tz], ...]
  - cx, cy: principal point at input_size scale
  - input_size: 0.5 (half resolution)
- split_info.json:
  - split_num: int
  - split: [[frame_indices], ...] 
- fps.txt: "FPS: 24.0\nProcessing time: ..."
- Full resolution: 1280x720
"""

import argparse
import os
import json
import re
import glob
import numpy as np
from dataclasses import dataclass, field
import subprocess
import sys
import shutil
import time
import copy
import csv

import torch
from pathlib import Path
from dpvo.config import cfg as dpvo_cfg
from dpvo.dpvo import DPVO
from dpvo.stream import image_stream
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

from multiprocessing import Process, Queue
from evo.core import metrics, sync, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.tools import file_interface
from evo.core import lie_algebra


@dataclass
class EvalConfig:
    scene_ids: list[str]
    omniworld_root: str             # path containing scene_id folders
    output_root: str = None         # where results go; defaults to ./outputs
    dpvo_model: str = ""            # path to model weights (optional, uses DPVO default)
    stride: int = 1                 # frame stride for DPVO
    image_width: int = 1280         # resolution of Omniworld-Game
    image_height: int = 720
    default_fps: float = 24.0

    def __post_init__(self):
        
        # set defaults for DPVO directory and output directory
        self.dpvo_dir = os.path.dirname(os.path.realpath(__file__))
        if self.output_root is None:
            self.output_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'outputs')

        #convert paths to absolutes
        self.dpvo_dir = os.path.abspath(self.dpvo_dir)
        self.omniworld_root = os.path.abspath(self.omniworld_root)
        self.output_root = os.path.abspath(self.output_root)
        if self.dpvo_model:
            self.dpvo_model = os.path.abspath(self.dpvo_model) 

    def scene_dir(self, scene_id: str) -> str:
        return os.path.join(self.omniworld_root, scene_id)

    def scene_output_dir(self, scene_id: str) -> str:
        return os.path.join(self.output_root, scene_id)

    def split_output_dir(self, scene_id: str, split_idx: int) -> str:
        return os.path.join(self.scene_output_dir(scene_id), f"split_{split_idx}")

    def demo_script(self) -> str:
        return os.path.join(self.dpvo_dir, "demo.py")

    def validate(self):
        """Check that required paths exist."""
        errors = []
        if not os.path.isfile(self.demo_script()):
            errors.append(f"DPVO demo.py not found: {self.demo_script()}")
        for sid in self.scene_ids:
            sd = self.scene_dir(sid)
            if not os.path.isdir(sd):
                errors.append(f"Scene directory not found: {sd}")
            elif not os.path.isdir(os.path.join(sd, "color")):
                errors.append(f"color/ not found in {sd}")
            elif not os.path.isdir(os.path.join(sd, "camera")):
                errors.append(f"camera/ not found in {sd}")
        if errors:
            raise FileNotFoundError("\n".join(errors))


@dataclass
class SceneInfo:
    """Parsed metadata for one scene."""
    scene_id: str
    split_num: int
    splits: list[list[int]]   # splits[i] = list of frame indices
    fps: float

    @staticmethod
    def load(scene_dir: str, default_fps: float = 24.0) -> "SceneInfo":
        scene_id = os.path.basename(scene_dir)

        # Parse split_info.json
        split_info_path = os.path.join(scene_dir, "split_info.json")
        with open(split_info_path, "r") as f:
            info = json.load(f)
        split_num = info["split_num"]
        splits = info["split"]

        assert len(splits) == split_num, (
            f"split_num={split_num} but got {len(splits)} splits"
        )

        # Parse fps.txt
        fps = default_fps
        fps_path = os.path.join(scene_dir, "fps.txt")
        if os.path.exists(fps_path):
            with open(fps_path, "r") as f:
                match = re.search(r"[\d.]+", f.read())
                if match:
                    fps = float(match.group())

        return SceneInfo(
            scene_id=scene_id,
            split_num=split_num,
            splits=splits,
            fps=fps,
        )


@dataclass
class PreparedSplit:
    """Paths to prepared data for one split, ready for DPVO."""
    calib_path: str
    gt_tum_path: str
    images_dir: str
    est_tum_path: str      # where DPVO output will go
    results_dir: str       # where evo results will go
    num_frames: int


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(
        description="Evaluate DPVO on OmniWorld-Game scenes"
    )
    parser.add_argument(
        "--scenes", nargs="+", required=True,
        help="Scene IDs to evaluate (e.g., 1f79eb96f021 b04f88d1f85a)"
    )
    parser.add_argument(
        "--omniworld_root", required=True,
        help="Path to directory containing scene folders"
    )
    parser.add_argument(
        "--output_root",
        help="Path for evaluation outputs"
    )
    parser.add_argument(
        "--dpvo_model", default="",
        help="Path to DPVO model weights (optional)"
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Frame stride for DPVO"
    )
    args = parser.parse_args()

    config = EvalConfig(
        scene_ids=args.scenes,
        omniworld_root=args.omniworld_root,
        output_root=args.output_root,
        dpvo_model=args.dpvo_model,
        stride=args.stride,
    )
    config.validate()
    return config


def prepare_split(
    config: EvalConfig,
    scene_info: SceneInfo,
    split_idx: int,
) -> PreparedSplit:
    """
    Convert one OmniWorld-Game split into DPVO-ready format.

    Reads camera/split_<N>.json, writes calib.txt and gt_tum.txt,
    symlinks the correct frames into a flat images/ directory.
    """
    scene_dir = config.scene_dir(scene_info.scene_id)
    out_dir = config.split_output_dir(scene_info.scene_id, split_idx)
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Load camera JSON
    camera_path = os.path.join(scene_dir, "camera", f"split_{split_idx}.json")
    with open(camera_path, "r") as f:
        cam = json.load(f)

    num_frames = len(cam["focals"])
    frame_indices = scene_info.splits[split_idx]
    assert len(frame_indices) == num_frames, (
        f"Split {split_idx}: split_info has {len(frame_indices)} frames "
        f"but camera JSON has {num_frames}"
    )

    # Write calib.txt
    input_size = cam.get("input_size", 1.0)
    scale = 1.0 / input_size
    focal = float(np.median(cam["focals"])) * scale
    cx = cam["cx"] * scale
    cy = cam["cy"] * scale

    calib_path = os.path.join(out_dir, "calib.txt")
    with open(calib_path, "w") as f:
        f.write(f"{focal} {focal} {cx} {cy}\n")

    # --- Write gt_tum.txt ---
    quats = np.array(cam["quats"])   # (N, 4) as [w, x, y, z]
    trans = np.array(cam["trans"])   # (N, 3)

    # Normalize quaternions
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / norms

    # Ensure consistent sign (w > 0)
    sign = np.sign(quats[:, 0:1])
    sign[sign == 0] = 1
    quats = quats * sign

    gt_tum_path = os.path.join(out_dir, "gt_tum.txt")
    with open(gt_tum_path, "w") as f:
        for i in range(num_frames):
            timestamp = float(i)
            tx, ty, tz = trans[i]
            w, x, y, z = quats[i]
            # TUM format: timestamp tx ty tz qx qy qz qw
            f.write(
                f"{timestamp:.6f} {tx:.8f} {ty:.8f} {tz:.8f} "
                f"{x:.8f} {y:.8f} {z:.8f} {w:.8f}\n"
            )

    # --- Symlink images ---
    color_dir = os.path.join(scene_dir, "color")
    for i, frame_idx in enumerate(frame_indices):
        src = os.path.join(color_dir, f"{frame_idx:06d}.png")
        dst = os.path.join(images_dir, f"{i:06d}.png")
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)

    # --- Define output paths ---
    est_tum_path = os.path.join(out_dir, "est_tum.txt")
    results_dir = os.path.join(out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    return PreparedSplit(
        calib_path=os.path.abspath(calib_path),
        gt_tum_path=os.path.abspath(gt_tum_path),
        images_dir=os.path.abspath(images_dir),
        est_tum_path=os.path.abspath(est_tum_path),
        results_dir=os.path.abspath(results_dir),
        num_frames=num_frames,
    )


@torch.no_grad()
def run_dpvo(config: EvalConfig, prepared: PreparedSplit) -> bool:
    """
    Run DPVO on a prepared split by calling the DPVO pipeline directly.
    
    Returns True if successful and est_tum.txt was produced.
    """
    print(f"  Running DPVO on {prepared.num_frames} frames...")

    # Load DPVO config
    dpvo_cfg.merge_from_file(os.path.join(config.dpvo_dir, "config", "default.yaml"))

    network = config.dpvo_model if config.dpvo_model else os.path.join(config.dpvo_dir, "dpvo.pth")

    try:
        # Run DPVO
        slam = None
        queue = Queue(maxsize=8)
        reader = Process(
            target=image_stream,
            args=(queue, prepared.images_dir, prepared.calib_path, config.stride, 0),
        )
        reader.start()

        while True:
            (t, image, intrinsics) = queue.get()
            if t < 0:
                break

            image = torch.from_numpy(image).permute(2, 0, 1).cuda()
            intrinsics = torch.from_numpy(intrinsics).cuda()

            if slam is None:
                _, H, W = image.shape
                slam = DPVO(dpvo_cfg, network, ht=H, wd=W, viz=False)

            slam(t, image, intrinsics)

        reader.join()

        # Extract results
        (poses, tstamps) = slam.terminate()

        trajectory = PoseTrajectory3D(
            positions_xyz=poses[:, :3],
            orientations_quat_wxyz=poses[:, [6, 3, 4, 5]],
            timestamps=tstamps,
        )

        # Save trajectory
        file_interface.write_tum_trajectory_file(prepared.est_tum_path, trajectory)
        print(f"  Trajectory saved: {prepared.est_tum_path}")
        return True

    except Exception as e:
        print(f"  DPVO failed: {e}")
        import traceback
        traceback.print_exc()
        return False

@dataclass
class SplitResults:
    """Evaluation metrics for one split."""
    scene_id: str
    split_idx: int
    num_frames: int
    ate_rmse: float = float("nan")
    ate_mean: float = float("nan")
    ate_median: float = float("nan")
    rpe_trans_rmse: float = float("nan")
    rpe_trans_mean: float = float("nan")
    rpe_rot_rmse: float = float("nan")     # degrees
    rpe_rot_mean: float = float("nan")     # degrees
    success: bool = False


def evaluate_split(prepared: PreparedSplit, scene_id: str, split_idx: int) -> SplitResults:
    """
    Evaluate estimated trajectory against ground truth using evo.

    Computes:
    - ATE (RMSE, mean, median) with Sim(3) alignment
    - RPE translation (RMSE, mean) with Sim(3) alignment, delta=1 frame
    - RPE rotation in degrees (RMSE, mean) with Sim(3) alignment, delta=1 frame

    Returns a SplitResults dataclass.
    """
    result = SplitResults(
        scene_id=scene_id,
        split_idx=split_idx,
        num_frames=prepared.num_frames,
    )

    if not os.path.exists(prepared.est_tum_path):
        print(f"  No estimated trajectory found, skipping evaluation.")
        return result

    # Load trajectories
    try:
        traj_gt = file_interface.read_tum_trajectory_file(prepared.gt_tum_path)
        traj_est = file_interface.read_tum_trajectory_file(prepared.est_tum_path)
    except Exception as e:
        print(f"  Failed to load trajectories: {e}")
        return result

    # Synchronize timestamps
    traj_gt, traj_est = sync.associate_trajectories(traj_gt, traj_est)

    if len(traj_est.timestamps) < 2:
        print(f"  Too few matched poses ({len(traj_est.timestamps)}), skipping.")
        return result

    # --- ATE ---
    traj_est_aligned = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_gt, correct_scale=True)

    ate_metric = metrics.APE(PoseRelation.translation_part)
    ate_metric.process_data((traj_gt, traj_est_aligned))
    ate_stats = ate_metric.get_all_statistics()

    result.ate_rmse = ate_stats["rmse"]
    result.ate_mean = ate_stats["mean"]
    result.ate_median = ate_stats["median"]

    # --- RPE translation ---
    rpe_trans_metric = metrics.RPE(
        PoseRelation.translation_part,
        delta=1,
        delta_unit=Unit.frames,
    )
    traj_est_rpe = copy.deepcopy(traj_est)
    traj_est_rpe.align(traj_gt, correct_scale=True)
    rpe_trans_metric.process_data((traj_gt, traj_est_rpe))
    rpe_trans_stats = rpe_trans_metric.get_all_statistics()

    result.rpe_trans_rmse = rpe_trans_stats["rmse"]
    result.rpe_trans_mean = rpe_trans_stats["mean"]

    # --- RPE rotation ---
    rpe_rot_metric = metrics.RPE(
        PoseRelation.rotation_angle_deg,
        delta=1,
        delta_unit=Unit.frames,
    )
    traj_est_rot = copy.deepcopy(traj_est)
    traj_est_rot.align(traj_gt, correct_scale=True)
    rpe_rot_metric.process_data((traj_gt, traj_est_rot))
    rpe_rot_stats = rpe_rot_metric.get_all_statistics()

    result.rpe_rot_rmse = rpe_rot_stats["rmse"]
    result.rpe_rot_mean = rpe_rot_stats["mean"]

    result.success = True

    print(f"  ATE RMSE: {result.ate_rmse:.4f} | "
          f"RPE trans: {result.rpe_trans_rmse:.4f} | "
          f"RPE rot: {result.rpe_rot_rmse:.4f}°")

    return result


import csv


@dataclass
class SceneResults:
    """Averaged metrics across all splits of one scene."""
    scene_id: str
    num_splits: int
    num_successful: int
    ate_rmse: float
    ate_mean: float
    ate_median: float
    rpe_trans_rmse: float
    rpe_trans_mean: float
    rpe_rot_rmse: float
    rpe_rot_mean: float

    @staticmethod
    def aggregate(scene_id: str, split_results: list[SplitResults]) -> "SceneResults":
        successful = [r for r in split_results if r.success]
        n = len(successful)
        if n == 0:
            return SceneResults(
                scene_id=scene_id,
                num_splits=len(split_results),
                num_successful=0,
                ate_rmse=float("nan"),
                ate_mean=float("nan"),
                ate_median=float("nan"),
                rpe_trans_rmse=float("nan"),
                rpe_trans_mean=float("nan"),
                rpe_rot_rmse=float("nan"),
                rpe_rot_mean=float("nan"),
            )
        return SceneResults(
            scene_id=scene_id,
            num_splits=len(split_results),
            num_successful=n,
            ate_rmse=sum(r.ate_rmse for r in successful) / n,
            ate_mean=sum(r.ate_mean for r in successful) / n,
            ate_median=sum(r.ate_median for r in successful) / n,
            rpe_trans_rmse=sum(r.rpe_trans_rmse for r in successful) / n,
            rpe_trans_mean=sum(r.rpe_trans_mean for r in successful) / n,
            rpe_rot_rmse=sum(r.rpe_rot_rmse for r in successful) / n,
            rpe_rot_mean=sum(r.rpe_rot_mean for r in successful) / n,
        )


def write_split_csv(path: str, all_split_results: list[SplitResults]):
    """Write per-split results to CSV."""
    fieldnames = [
        "scene_id", "split_idx", "num_frames", "success",
        "ate_rmse", "ate_mean", "ate_median",
        "rpe_trans_rmse", "rpe_trans_mean",
        "rpe_rot_rmse", "rpe_rot_mean",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_split_results:
            writer.writerow({
                "scene_id": r.scene_id,
                "split_idx": r.split_idx,
                "num_frames": r.num_frames,
                "success": r.success,
                "ate_rmse": f"{r.ate_rmse:.6f}" if r.success else "",
                "ate_mean": f"{r.ate_mean:.6f}" if r.success else "",
                "ate_median": f"{r.ate_median:.6f}" if r.success else "",
                "rpe_trans_rmse": f"{r.rpe_trans_rmse:.6f}" if r.success else "",
                "rpe_trans_mean": f"{r.rpe_trans_mean:.6f}" if r.success else "",
                "rpe_rot_rmse": f"{r.rpe_rot_rmse:.6f}" if r.success else "",
                "rpe_rot_mean": f"{r.rpe_rot_mean:.6f}" if r.success else "",
            })


def write_scene_csv(path: str, scene_results: list[SceneResults]):
    """Write per-scene averaged results to CSV."""
    fieldnames = [
        "scene_id", "num_splits", "num_successful",
        "ate_rmse", "ate_mean", "ate_median",
        "rpe_trans_rmse", "rpe_trans_mean",
        "rpe_rot_rmse", "rpe_rot_mean",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in scene_results:
            writer.writerow({
                "scene_id": r.scene_id,
                "num_splits": r.num_splits,
                "num_successful": r.num_successful,
                "ate_rmse": f"{r.ate_rmse:.6f}" if r.num_successful > 0 else "",
                "ate_mean": f"{r.ate_mean:.6f}" if r.num_successful > 0 else "",
                "ate_median": f"{r.ate_median:.6f}" if r.num_successful > 0 else "",
                "rpe_trans_rmse": f"{r.rpe_trans_rmse:.6f}" if r.num_successful > 0 else "",
                "rpe_trans_mean": f"{r.rpe_trans_mean:.6f}" if r.num_successful > 0 else "",
                "rpe_rot_rmse": f"{r.rpe_rot_rmse:.6f}" if r.num_successful > 0 else "",
                "rpe_rot_mean": f"{r.rpe_rot_mean:.6f}" if r.num_successful > 0 else "",
            })


def print_summary(scene_results: list[SceneResults]):
    """Print a formatted summary table."""
    header = (
        f"{'Scene':<20} {'Splits':>7} {'OK':>4} "
        f"{'ATE RMSE':>10} {'RPE trans':>10} {'RPE rot':>10}"
    )
    print("\n" + "=" * len(header))
    print("EVALUATION SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in scene_results:
        if r.num_successful > 0:
            print(
                f"{r.scene_id:<20} {r.num_splits:>7} {r.num_successful:>4} "
                f"{r.ate_rmse:>10.4f} {r.rpe_trans_rmse:>10.4f} {r.rpe_rot_rmse:>9.4f}°"
            )
        else:
            print(
                f"{r.scene_id:<20} {r.num_splits:>7} {r.num_successful:>4} "
                f"{'FAILED':>10} {'':>10} {'':>10}"
            )
    print("=" * len(header))


def main():
    config = parse_args()
    os.makedirs(config.output_root, exist_ok=True)

    all_split_results = []
    all_scene_results = []

    for scene_id in config.scene_ids:
        print(f"\n{'='*60}")
        print(f"Scene: {scene_id}")
        print(f"{'='*60}")

        scene_dir = config.scene_dir(scene_id)
        scene_info = SceneInfo.load(scene_dir, config.default_fps)
        print(f"  Splits: {scene_info.split_num} | FPS: {scene_info.fps}")

        scene_split_results = []

        for split_idx in range(scene_info.split_num):
            print(f"\n  --- Split {split_idx}/{scene_info.split_num - 1} ---")

            # Prepare
            prepared = prepare_split(config, scene_info, split_idx)
            print(f"  Prepared: {prepared.num_frames} frames")

            # Run DPVO
            success = run_dpvo(config, prepared)

            # Evaluate
            if success:
                result = evaluate_split(prepared, scene_id, split_idx)
            else:
                result = SplitResults(
                    scene_id=scene_id,
                    split_idx=split_idx,
                    num_frames=prepared.num_frames,
                )

            scene_split_results.append(result)
            all_split_results.append(result)

        # Aggregate scene results
        scene_result = SceneResults.aggregate(scene_id, scene_split_results)
        all_scene_results.append(scene_result)

        # Write per-scene split CSV
        scene_csv = os.path.join(config.scene_output_dir(scene_id), "splits.csv")
        os.makedirs(os.path.dirname(scene_csv), exist_ok=True)
        write_split_csv(scene_csv, scene_split_results)
        print(f"\n  Per-split results: {scene_csv}")

    # Write aggregated CSVs
    write_split_csv(os.path.join(config.output_root, "all_splits.csv"), all_split_results)
    write_scene_csv(os.path.join(config.output_root, "all_scenes.csv"), all_scene_results)

    # Print summary
    print_summary(all_scene_results)

    print(f"\nAll results saved to {config.output_root}/")


if __name__ == "__main__":
    main()
