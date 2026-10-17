"""Run OpenCap's pose-detection step on one camera's video, on CPU or GPU.

    python biostrike/pose_mmpose.py --video <Cam>/InputMedia/<trial>/<trial>.mov \
        --weights <dir with the two .pth files> [--device cpu|cuda:0] [--max-frames N]

This is OpenCap's own code path, not a reimplementation: the same
`utilsMMpose.detection_inference` / `pose_inference` that OpenCap's MMPose
container runs (mmpose/loop_mmpose.py), with the same configs, weights and
person-box threshold (defaultOpenCapSettings.json, 0.8). Only the device is a
parameter, so it runs without an NVIDIA GPU.

Outputs land where main.py looks for them, so the rest of the pipeline then
runs without pose detection:
    <Cam>/OutputPkl_mmpose_0.8/<trial>/<trial>_rotated.pkl      raw MMPose results
    <Cam>/OutputPkl_mmpose_0.8/<trial>/<trial>_rotated_pp.pkl   OpenPose-layout keypoints

Needs OpenCap's pose stack (MMPose 0.28.1, MMDetection 2.25.1, mmcv-full
1.3.17, PyTorch 1.9). See biostrike/requirements-pose.txt.
"""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MMPOSE_DIR = os.path.join(REPO, 'mmpose')
PERSON_CONFIG = 'faster_rcnn_r50_fpn_coco.py'
PERSON_WEIGHTS = 'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
POSE_CONFIG = 'hrnet_w48_coco_wholebody_384x288_dark_plus.py'
POSE_WEIGHTS = 'hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth'


def bbox_threshold() -> float:
    with open(os.path.join(REPO, 'defaultOpenCapSettings.json')) as f:
        return float(json.load(f)['hrnet'])


def rotated_video(video: str, max_frames: int = 0) -> str:
    """The unrotated copy OpenCap runs detection on (utilsDetector.runMMposeVideo):
    `ffmpeg -i <video> -q 0 <trial>_rotated.avi`, which applies the phone's
    orientation tag."""
    out = os.path.splitext(video)[0] + '_rotated.avi'
    # Always rebuilt: a copy left by a --max-frames test would silently
    # truncate a full run.
    cmd = ['ffmpeg', '-loglevel', 'error', '-y', '-i', video]
    if max_frames:
        cmd += ['-frames:v', str(max_frames)]
    subprocess.run(cmd + ['-q', '0', out], check=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--video', required=True)
    ap.add_argument('--weights', required=True, help='folder holding the two .pth files')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-frames', type=int, default=0, help='only the first N frames (timing tests)')
    args = ap.parse_args(argv)

    # OpenCap's utilsMMpose.py sits in the repo root and its helpers in
    # <repo>/mmpose. Append, never prepend: that folder shares its name with
    # the installed mmpose package, which must stay first on the path.
    sys.path.append(MMPOSE_DIR)
    sys.path.append(REPO)
    from utilsMMpose import detection_inference, pose_inference
    import mmpose
    assert 'site-packages' in mmpose.__file__, mmpose.__file__

    video = os.path.abspath(args.video)
    trial = os.path.basename(os.path.dirname(video))
    cam_dir = os.path.dirname(os.path.dirname(os.path.dirname(video)))
    thr = bbox_threshold()
    out_dir = os.path.join(cam_dir, f'OutputPkl_mmpose_{thr}', trial)
    box_dir = os.path.join(cam_dir, 'OutputBox_mmpose', trial)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(box_dir, exist_ok=True)

    avi = rotated_video(video, args.max_frames)
    prefix = os.path.splitext(os.path.basename(avi))[0]
    box_pkl = os.path.join(box_dir, prefix + '.pkl')
    raw_pkl = os.path.join(out_dir, prefix + '.pkl')

    t0 = time.time()
    detection_inference(os.path.join(MMPOSE_DIR, PERSON_CONFIG),
                        os.path.join(args.weights, PERSON_WEIGHTS), avi, box_pkl,
                        device=args.device)
    t1 = time.time()
    pose_inference(os.path.join(MMPOSE_DIR, POSE_CONFIG),
                   os.path.join(args.weights, POSE_WEIGHTS), avi, box_pkl, raw_pkl, '',
                   device=args.device, batch_size=args.batch_size, bbox_thr=thr,
                   visualize=False)
    t2 = time.time()
    with open(box_pkl, 'rb') as f:
        n = len(pickle.load(f))
    print(f'{trial}: {n} frames · person detection {t1 - t0:.0f} s ({(t1 - t0) / n:.2f} s/frame) · '
          f'pose {t2 - t1:.0f} s ({(t2 - t1) / n:.2f} s/frame)')
    print('raw keypoints:', raw_pkl)
    return 0


if __name__ == '__main__':
    sys.exit(main())
