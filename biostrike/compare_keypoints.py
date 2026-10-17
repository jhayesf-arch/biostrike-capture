"""Compare 2-D keypoints from our pose run against OpenCap's, frame by frame.

    python biostrike/compare_keypoints.py OURS_pp.pkl OPENCAP_keypoints.pkl [--json report.json]

Both files use OpenCap's post-processed layout: per frame, a list of people,
each {'pose_keypoints_2d': [x, y, confidence] * 25}. OpenCap's own file is
<Cam>/OutputPkl/<trial>_keypoints.pkl in a session download; ours is
<Cam>/OutputPkl_mmpose_0.8/<trial>/<trial>_rotated_pp.pkl.

For each frame the athlete is the person nearest the previous frame's match
(the first frame takes the most confident person), in both files. Errors are
pixel distances in the image, counted only where both detectors are at least
somewhat confident (default 0.3, the threshold OpenCap's own visualization
uses).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from capture_local import _is_plain_data_pickle  # noqa: E402

OPENPOSE_25 = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist',
               'midHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye',
               'REar', 'LEar', 'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel']


def load_pp(path: str) -> list:
    if not _is_plain_data_pickle(path):
        raise ValueError(f'{path} contains executable pickle opcodes; refusing to load it.')
    with open(path, 'rb') as f:
        return pickle.load(f)


def track_athlete(frames: list) -> np.ndarray:
    """(n_frames, 25, 3) for one person, NaN where nobody was found."""
    out = np.full((len(frames), 25, 3), np.nan)
    prev = None
    for i, people in enumerate(frames):
        cands = [np.asarray(p['pose_keypoints_2d'], float).reshape(25, 3) for p in people]
        if not cands:
            continue
        if prev is None:
            pick = max(cands, key=lambda k: np.nanmean(k[:, 2]))
        else:
            pick = min(cands, key=lambda k: np.nanmean(np.linalg.norm(k[:, :2] - prev[:, :2], axis=1)))
        out[i] = pick
        prev = pick
    return out


def compare(ours: str, ref: str, conf: float = 0.3) -> dict:
    a, b = track_athlete(load_pp(ours)), track_athlete(load_pp(ref))
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    both = (a[:, :, 2] >= conf) & (b[:, :, 2] >= conf)
    dist = np.linalg.norm(a[:, :, :2] - b[:, :, :2], axis=2)
    dist[~both] = np.nan
    per = {}
    for j, name in enumerate(OPENPOSE_25):
        d = dist[:, j][np.isfinite(dist[:, j])]
        if d.size:
            per[name] = {'median_px': float(np.median(d)), 'p95_px': float(np.percentile(d, 95)),
                         'max_px': float(d.max()), 'frames': int(d.size)}
    allv = dist[np.isfinite(dist)]
    conf_diff = np.abs(a[:, :, 2] - b[:, :, 2])
    return {
        'ours': ours, 'reference': ref, 'frames_ours': len(load_pp(ours)),
        'frames_reference': len(load_pp(ref)), 'frames_compared': n,
        'points_compared': int(allv.size),
        'median_px': float(np.median(allv)) if allv.size else None,
        'p95_px': float(np.percentile(allv, 95)) if allv.size else None,
        'max_px': float(allv.max()) if allv.size else None,
        'median_confidence_diff': float(np.nanmedian(conf_diff)),
        'per_keypoint': per,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('ours')
    ap.add_argument('reference')
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--json')
    args = ap.parse_args(argv)
    r = compare(args.ours, args.reference, args.conf)
    print(f"frames: ours {r['frames_ours']}, OpenCap {r['frames_reference']}, compared {r['frames_compared']}")
    print(f"all points: median {r['median_px']:.2f} px · 95th pct {r['p95_px']:.2f} px · "
          f"max {r['max_px']:.1f} px over {r['points_compared']} points")
    print(f"confidence: median difference {r['median_confidence_diff']:.4f}")
    worst = sorted(r['per_keypoint'].items(), key=lambda kv: -kv[1]['p95_px'])[:6]
    for name, v in worst:
        print(f"  {name:10s} median {v['median_px']:.2f} px · 95th {v['p95_px']:.2f} px · max {v['max_px']:.1f} px")
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(r, f, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
