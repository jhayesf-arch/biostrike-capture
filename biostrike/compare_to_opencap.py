"""Compare this fork's pipeline output against OpenCap's for the same trial.

This is the acceptance test for running the OpenCap pipeline ourselves: same
session, same videos — do we reproduce Stanford's numbers?

    python biostrike/compare_to_opencap.py OURS.trc REFERENCE.trc
    python biostrike/compare_to_opencap.py OURS.mot REFERENCE.mot --json report.json

.trc (marker positions): per-marker 3-D RMSE in millimetres, reported
separately for video keypoints and for the augmented anatomical markers
(names ending _study), since the augmenter is a distinct stage that can
diverge on its own.

.mot (joint coordinates): per-coordinate RMSE — degrees for rotations,
millimetres for the pelvis translations.

Both files are put on a common time base before comparing (the overlapping
window, linearly interpolated), so differing start frames or lengths do not
inflate the error.

Exit status is 0 on pass and 1 on fail, so it can gate a script or CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Defaults sized for "same code, same weights": a faithful reproduction should
# sit far below these. Library-version float differences show up as fractions
# of a millimetre; a wrong calibration, sync or augmenter shows up as tens.
DEFAULT_MARKER_TOL_MM = 10.0
DEFAULT_ANGLE_TOL_DEG = 2.0
DEFAULT_TRANSLATION_TOL_MM = 10.0
PELVIS_TRANSLATIONS = ('pelvis_tx', 'pelvis_ty', 'pelvis_tz')


# ── Readers ────────────────────────────────────────────────────────────────

def read_trc(path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray], str]:
    """Return (time, {marker: (N,3) array}, units) from an OpenSim .trc file."""
    with open(path) as f:
        lines = f.read().splitlines()
    meta_keys = lines[1].split('\t')
    meta_vals = lines[2].split('\t')
    meta = dict(zip(meta_keys, meta_vals))
    units = meta.get('Units', 'm').strip()

    names = [n.strip() for n in lines[3].split('\t')[2:] if n.strip()]
    rows = [l.split('\t') for l in lines[5:] if l.strip()]
    width = 2 + 3 * len(names)
    data = np.array([[float(x) if x.strip() else np.nan for x in (r + [''] * width)[:width]]
                     for r in rows])
    time = data[:, 1]
    markers = {n: data[:, 2 + 3 * i: 5 + 3 * i] for i, n in enumerate(names)}
    return time, markers, units


def read_mot(path: str) -> pd.DataFrame:
    """Return a DataFrame from an OpenSim .mot/.sto file (header skipped)."""
    with open(path) as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if l.strip().lower() == 'endheader') + 1
    df = pd.read_csv(path, sep=r'\s+', skiprows=start)
    df.columns = [c.strip() for c in df.columns]
    return df


def mot_in_degrees(path: str) -> bool:
    with open(path) as f:
        for line in f:
            if line.strip().lower() == 'endheader':
                break
            if line.lower().startswith('indegrees'):
                return 'yes' in line.lower()
    return True


# ── Alignment ──────────────────────────────────────────────────────────────

def _common_time(t_a: np.ndarray, t_b: np.ndarray) -> np.ndarray:
    """Overlapping window on the finer of the two sampling grids."""
    lo, hi = max(t_a.min(), t_b.min()), min(t_a.max(), t_b.max())
    if hi <= lo:
        raise ValueError(f'no time overlap: [{t_a.min():.3f}, {t_a.max():.3f}] vs '
                         f'[{t_b.min():.3f}, {t_b.max():.3f}]')
    dt = min(np.median(np.diff(t_a)), np.median(np.diff(t_b)))
    return np.arange(lo, hi + dt / 2, dt)


def _interp(t_src: np.ndarray, y: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return np.interp(t_dst, t_src, y)
    return np.column_stack([np.interp(t_dst, t_src, y[:, j]) for j in range(y.shape[1])])


# ── Comparisons ────────────────────────────────────────────────────────────

def compare_trc(ours: str, ref: str, tol_mm: float = DEFAULT_MARKER_TOL_MM) -> Dict:
    t_o, m_o, u_o = read_trc(ours)
    t_r, m_r, u_r = read_trc(ref)
    scale = {'m': 1000.0, 'mm': 1.0}
    if u_o not in scale or u_r not in scale:
        raise ValueError(f'unsupported units: ours={u_o!r} ref={u_r!r}')
    t = _common_time(t_o, t_r)

    shared = [n for n in m_r if n in m_o]
    missing = sorted(set(m_r) - set(m_o))
    per_marker = {}
    for n in shared:
        a = _interp(t_o, m_o[n] * scale[u_o], t)
        b = _interp(t_r, m_r[n] * scale[u_r], t)
        err = np.linalg.norm(a - b, axis=1)
        err = err[np.isfinite(err)]
        if err.size:
            per_marker[n] = {'rmse_mm': float(np.sqrt(np.mean(err ** 2))),
                             'max_mm': float(np.max(err))}

    def group(names: List[str]) -> Dict:
        vals = [per_marker[n]['rmse_mm'] for n in names if n in per_marker]
        if not vals:
            return {'n': 0}
        return {'n': len(vals), 'median_rmse_mm': float(np.median(vals)),
                'worst_rmse_mm': float(np.max(vals)),
                'worst_marker': max((n for n in names if n in per_marker),
                                    key=lambda n: per_marker[n]['rmse_mm'])}

    anatomical = [n for n in per_marker if n.endswith('_study')]
    keypoints = [n for n in per_marker if not n.endswith('_study')]
    worst = max((v['rmse_mm'] for v in per_marker.values()), default=np.inf)
    return {
        'kind': 'trc', 'ours': ours, 'reference': ref,
        'overlap_s': [float(t[0]), float(t[-1])], 'frames_compared': int(len(t)),
        'markers_compared': len(per_marker), 'markers_missing_in_ours': missing,
        'video_keypoints': group(keypoints), 'anatomical_markers': group(anatomical),
        'per_marker': per_marker, 'tolerance_mm': tol_mm,
        'passed': bool(per_marker) and not missing and worst <= tol_mm,
    }


def compare_mot(ours: str, ref: str,
                tol_deg: float = DEFAULT_ANGLE_TOL_DEG,
                tol_mm: float = DEFAULT_TRANSLATION_TOL_MM) -> Dict:
    a, b = read_mot(ours), read_mot(ref)
    deg_a, deg_b = mot_in_degrees(ours), mot_in_degrees(ref)
    t = _common_time(a['time'].values, b['time'].values)

    shared = [c for c in b.columns if c != 'time' and c in a.columns]
    missing = sorted(set(b.columns) - set(a.columns) - {'time'})
    per_coord, failures = {}, []
    for c in shared:
        ya, yb = a[c].values.astype(float), b[c].values.astype(float)
        if c in PELVIS_TRANSLATIONS:
            ya, yb, unit, tol = ya * 1000.0, yb * 1000.0, 'mm', tol_mm
        else:
            if not deg_a:
                ya = np.degrees(ya)
            if not deg_b:
                yb = np.degrees(yb)
            unit, tol = 'deg', tol_deg
        d = _interp(a['time'].values, ya, t) - _interp(b['time'].values, yb, t)
        d = d[np.isfinite(d)]
        rmse = float(np.sqrt(np.mean(d ** 2))) if d.size else np.inf
        per_coord[c] = {'rmse': rmse, 'max': float(np.max(np.abs(d))) if d.size else np.inf,
                        'unit': unit}
        if rmse > tol:
            failures.append(c)

    rot = [v['rmse'] for c, v in per_coord.items() if v['unit'] == 'deg']
    return {
        'kind': 'mot', 'ours': ours, 'reference': ref,
        'overlap_s': [float(t[0]), float(t[-1])], 'frames_compared': int(len(t)),
        'coordinates_compared': len(per_coord), 'coordinates_missing_in_ours': missing,
        'median_rotation_rmse_deg': float(np.median(rot)) if rot else None,
        'failing_coordinates': failures, 'per_coordinate': per_coord,
        'tolerance_deg': tol_deg, 'tolerance_mm': tol_mm,
        'passed': bool(per_coord) and not missing and not failures,
    }


# ── CLI ────────────────────────────────────────────────────────────────────

def _print_report(r: Dict) -> None:
    verdict = 'PASS' if r['passed'] else 'FAIL'
    print(f"{verdict}  {r['kind'].upper()}  ours={r['ours']}")
    print(f"      reference={r['reference']}")
    print(f"      compared {r['frames_compared']} frames over "
          f"{r['overlap_s'][0]:.3f}–{r['overlap_s'][1]:.3f} s")
    if r['kind'] == 'trc':
        for label in ('video_keypoints', 'anatomical_markers'):
            g = r[label]
            if g.get('n'):
                print(f"      {label:20s} n={g['n']:3d}  median RMSE {g['median_rmse_mm']:7.2f} mm"
                      f"  worst {g['worst_rmse_mm']:7.2f} mm ({g['worst_marker']})")
        if r['markers_missing_in_ours']:
            print(f"      missing in ours: {', '.join(r['markers_missing_in_ours'][:10])}")
        print(f"      tolerance {r['tolerance_mm']} mm per marker")
    else:
        if r['median_rotation_rmse_deg'] is not None:
            print(f"      median rotation RMSE {r['median_rotation_rmse_deg']:.3f} deg "
                  f"over {r['coordinates_compared']} coordinates")
        worst = sorted(r['per_coordinate'].items(), key=lambda kv: -kv[1]['rmse'])[:5]
        for c, v in worst:
            print(f"      {c:24s} RMSE {v['rmse']:8.3f} {v['unit']}")
        if r['coordinates_missing_in_ours']:
            print(f"      missing in ours: {', '.join(r['coordinates_missing_in_ours'][:10])}")
        print(f"      tolerance {r['tolerance_deg']} deg / {r['tolerance_mm']} mm")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('ours')
    ap.add_argument('reference')
    ap.add_argument('--marker-tol-mm', type=float, default=DEFAULT_MARKER_TOL_MM)
    ap.add_argument('--angle-tol-deg', type=float, default=DEFAULT_ANGLE_TOL_DEG)
    ap.add_argument('--translation-tol-mm', type=float, default=DEFAULT_TRANSLATION_TOL_MM)
    ap.add_argument('--json', help='also write the full report here')
    args = ap.parse_args(argv)

    if args.ours.lower().endswith('.trc'):
        report = compare_trc(args.ours, args.reference, args.marker_tol_mm)
    else:
        report = compare_mot(args.ours, args.reference, args.angle_tol_deg,
                             args.translation_tol_mm)
    _print_report(report)
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(report, f, indent=1)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
