"""Tests for compare_to_opencap. Run: python -m pytest biostrike  (or run this file)."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_to_opencap import compare_mot, compare_trc  # noqa: E402


def _write_trc(path, t, markers, units='m'):
    names = list(markers)
    with open(path, 'w') as f:
        f.write('PathFileType\t4\t(X/Y/Z)\ttest\n')
        f.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\t'
                'OrigDataStartFrame\tOrigNumFrames\n')
        f.write(f'60\t60\t{len(t)}\t{len(names)}\t{units}\t60\t1\t{len(t)}\n')
        f.write('Frame#\tTime\t' + '\t\t\t'.join(names) + '\n')
        f.write('\t\t' + '\t'.join(f'X{i}\tY{i}\tZ{i}' for i in range(1, len(names) + 1)) + '\n')
        for k, tk in enumerate(t):
            vals = '\t'.join('\t'.join(f'{v:.6f}' for v in markers[n][k]) for n in names)
            f.write(f'{k + 1}\t{tk:.6f}\t{vals}\n')


def _write_mot(path, t, cols, in_degrees=True):
    with open(path, 'w') as f:
        f.write(f'Coordinates\nversion=1\nnRows={len(t)}\nnColumns={len(cols) + 1}\n')
        f.write(f'inDegrees={"yes" if in_degrees else "no"}\nendheader\n')
        f.write('time\t' + '\t'.join(cols) + '\n')
        for k, tk in enumerate(t):
            f.write(f'{tk:.6f}\t' + '\t'.join(f'{cols[c][k]:.8f}' for c in cols) + '\n')


T = np.arange(0, 2, 1 / 60)
MARKERS = {
    'Neck': np.column_stack([np.sin(T), 1.5 + 0 * T, np.cos(T)]),
    'r.ASIS_study': np.column_stack([0.1 + 0 * T, 0.9 + 0.01 * T, 0.2 + 0 * T]),
}


def test_trc_identical_passes_at_zero():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'a.trc'); _write_trc(p, T, MARKERS)
        r = compare_trc(p, p)
    assert r['passed'] and r['video_keypoints']['median_rmse_mm'] == 0.0


def test_trc_known_offset_reported_in_mm():
    shifted = {k: v + np.array([0.003, 0.004, 0.0]) for k, v in MARKERS.items()}  # 5 mm
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, 'a.trc'), os.path.join(d, 'b.trc')
        _write_trc(a, T, shifted); _write_trc(b, T, MARKERS)
        r = compare_trc(a, b)
        assert abs(r['per_marker']['Neck']['rmse_mm'] - 5.0) < 1e-3
        assert r['passed']                              # 5 mm < 10 mm default
        assert not compare_trc(a, b, tol_mm=4.0)['passed']


def test_trc_units_mm_vs_m_and_time_offset():
    """A file in mm whose clip starts later must still line up exactly."""
    late = T[30:]
    in_mm = {k: v[30:] * 1000.0 for k, v in MARKERS.items()}
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, 'a.trc'), os.path.join(d, 'b.trc')
        _write_trc(a, late, in_mm, units='mm'); _write_trc(b, T, MARKERS)
        r = compare_trc(a, b)
    assert r['passed'] and r['per_marker']['Neck']['rmse_mm'] < 1e-2
    assert abs(r['overlap_s'][0] - late[0]) < 1e-9


def test_trc_missing_marker_fails():
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, 'a.trc'), os.path.join(d, 'b.trc')
        _write_trc(a, T, {'Neck': MARKERS['Neck']}); _write_trc(b, T, MARKERS)
        r = compare_trc(a, b)
    assert not r['passed'] and r['markers_missing_in_ours'] == ['r.ASIS_study']


def test_mot_degrees_radians_and_translation_units():
    knee = 30 * np.sin(T)
    ours = {'knee_angle_r': np.radians(knee + 1.0), 'pelvis_tx': 0.5 + 0.004 + 0 * T}
    ref = {'knee_angle_r': knee, 'pelvis_tx': 0.5 + 0 * T}
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, 'a.mot'), os.path.join(d, 'b.mot')
        _write_mot(a, T, ours, in_degrees=False); _write_mot(b, T, ref, in_degrees=True)
        r = compare_mot(a, b)
        assert abs(r['per_coordinate']['knee_angle_r']['rmse'] - 1.0) < 1e-6   # deg
        assert abs(r['per_coordinate']['pelvis_tx']['rmse'] - 4.0) < 1e-6      # mm
        assert r['passed']
        assert compare_mot(a, b, tol_deg=0.5)['failing_coordinates'] == ['knee_angle_r']


if __name__ == '__main__':
    tests = [v for k, v in dict(globals()).items() if k.startswith('test_')]
    for fn in tests:
        fn(); print(f'✓ {fn.__name__}')
    print(f'{len(tests)} passed')
