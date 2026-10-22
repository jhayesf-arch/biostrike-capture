"""Run pose detection on every camera and trial of a capture session.

    python biostrike/detect_session.py --session <session folder> --weights <dir>
        [--device cuda:0] [--trials swing_1 neutral] [--redo] [--dry-run]

Made for Google Colab (biostrike/colab/Biostrike_pose_GPU.ipynb), where the GPU
step runs and the rest of the pipeline stays on the Mac. Run it with the
interpreter that has OpenCap's pose stack (requirements-pose-gpu.txt).

For each <session>/Videos/Cam*/InputMedia/<trial>/<trial>.mov|.mp4 it calls
pose_mmpose.py, which leaves the raw MMPose keypoints at
    <Cam>/OutputPkl_mmpose_0.8/<trial>/<trial>_rotated.pkl
Trials already done are skipped, so a run cut off by a Colab disconnect picks
up where it stopped. Then, on the Mac:
    python biostrike/capture_local.py --session <session> --trial <trial> --keypoints-detected
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from typing import List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_EXTS = ('.mov', '.mp4', '.MOV', '.MP4')
CALIBRATION_TRIAL = 'calibration'
BBOX_THR = 0.8   # defaultOpenCapSettings.json 'hrnet'; folder name pose_mmpose.py writes to


def camera_dirs(session_dir: str) -> List[str]:
    return sorted(d for d in glob.glob(os.path.join(session_dir, 'Videos', 'Cam*')) if os.path.isdir(d))


def trial_video(cam_dir: str, trial: str) -> Optional[str]:
    for ext in VIDEO_EXTS:
        p = os.path.join(cam_dir, 'InputMedia', trial, trial + ext)
        if os.path.exists(p):
            return p
    return None


def trials_in(session_dir: str) -> List[str]:
    """Trials with a video in every camera folder (the checkerboard trial included)."""
    cams = camera_dirs(session_dir)
    if not cams:
        return []
    names = [os.path.basename(d) for d in glob.glob(os.path.join(cams[0], 'InputMedia', '*'))
             if os.path.isdir(d)]
    return sorted(t for t in names if all(trial_video(c, t) for c in cams))


def raw_keypoints(cam_dir: str, trial: str) -> str:
    return os.path.join(cam_dir, f'OutputPkl_mmpose_{BBOX_THR}', trial, f'{trial}_rotated.pkl')


def plan(session_dir: str, trials: Optional[List[str]] = None, redo: bool = False) -> List[tuple]:
    """[(cam_dir, trial, video, done)] for every camera x trial to consider."""
    out = []
    # The checkerboard trial is calibrated from the board, not from body keypoints.
    for trial in (trials or [t for t in trials_in(session_dir) if t != CALIBRATION_TRIAL]):
        for cam in camera_dirs(session_dir):
            video = trial_video(cam, trial)
            if video:
                done = os.path.exists(raw_keypoints(cam, trial)) and not redo
                out.append((cam, trial, video, done))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--session', required=True, help='session folder (has Videos/Cam0, Cam1, ...)')
    ap.add_argument('--weights', required=True, help='folder holding the two .pth files')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--trials', nargs='*', help='default: every trial with a video in each camera')
    ap.add_argument('--redo', action='store_true', help='run trials that already have keypoints')
    ap.add_argument('--dry-run', action='store_true', help='list the work and stop')
    ap.add_argument('--max-frames', type=int, default=0, help='only the first N frames (a quick test)')
    args = ap.parse_args(argv)

    session = os.path.abspath(args.session)
    cams = camera_dirs(session)
    if len(cams) < 2:
        print(f'Need at least 2 camera folders under {session}/Videos, found {len(cams)}.')
        return 2
    work = plan(session, args.trials, args.redo)
    if not work:
        print('No trial has a video in every camera folder.')
        return 2
    todo = [w for w in work if not w[3]]
    for cam, trial, video, done in work:
        print(f'{"done " if done else "to do"}  {os.path.basename(cam)}  {trial}')
    if args.dry_run or not todo:
        print(f'{len(todo)} of {len(work)} videos to process.')
        return 0

    t0 = time.time()
    for i, (cam, trial, video, _) in enumerate(todo, 1):
        print(f'\n[{i}/{len(todo)}] {os.path.basename(cam)} {trial} on {args.device}', flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, 'pose_mmpose.py'), '--video', video,
                        '--weights', args.weights, '--device', args.device]
                       + (['--max-frames', str(args.max_frames)] if args.max_frames else []), check=True)
    print(f'\nAll {len(todo)} videos done in {(time.time() - t0) / 60:.1f} min.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
