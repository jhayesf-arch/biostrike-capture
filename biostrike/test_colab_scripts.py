"""Tests for the Colab path: detect_session.py, process_session.py and
capture_local --keypoints-detected. No pose models, TensorFlow or OpenSim.
Run: python biostrike/test_colab_scripts.py"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capture_local as cl      # noqa: E402
import detect_session as ds     # noqa: E402
import process_session as ps    # noqa: E402


def _session(root, trials=('swing_1', 'neutral', 'calibration'), cams=2, scaled=False):
    s = os.path.join(root, 'session')
    os.makedirs(s)
    open(os.path.join(s, 'sessionMetadata.yaml'), 'w').write('openSimModel: LaiUhlrich2022\n')
    for i in range(cams):
        for t in trials:
            d = os.path.join(s, 'Videos', f'Cam{i}', 'InputMedia', t)
            os.makedirs(d)
            open(os.path.join(d, t + '.mov'), 'wb').write(b'v')
    if scaled:
        os.makedirs(os.path.join(s, 'OpenSimData', 'Model'))
        open(os.path.join(s, 'OpenSimData', 'Model', 'LaiUhlrich2022_scaled.osim'), 'w').write('<m/>')
    return s


def _detected(s, trial):
    for cam in ds.camera_dirs(s):
        p = ds.raw_keypoints(cam, trial)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, 'wb').write(b'k')


def test_detection_skips_the_checkerboard_and_finished_trials():
    root = tempfile.mkdtemp()
    try:
        s = _session(root)
        _detected(s, 'neutral')
        work = {(os.path.basename(c), t): done for c, t, _, done in ds.plan(s)}
        assert ('Cam0', 'calibration') not in work
        assert work[('Cam0', 'neutral')] is True and work[('Cam0', 'swing_1')] is False
        assert all(not d for d in (done for (_, _, _, done) in ds.plan(s, redo=True)))
    finally:
        shutil.rmtree(root)


def test_raw_keypoints_land_where_capture_local_reads_them():
    root = tempfile.mkdtemp()
    try:
        s = _session(root)
        cam = ds.camera_dirs(s)[0]
        assert ds.raw_keypoints(cam, 'swing_1') == cl.raw_path(cam, 'swing_1')
    finally:
        shutil.rmtree(root)


def test_preflight_asks_for_detection_before_the_cpu_steps():
    root = tempfile.mkdtemp()
    try:
        s = _session(root)
        problems = cl.preflight(s, 'swing_1', rescale=True, reuse_keypoints=False,
                                keypoints_detected=True)
        assert any('no detected keypoints for "swing_1"' in p for p in problems)
        assert any('no detected keypoints for "neutral"' in p for p in problems)
        _detected(s, 'swing_1')
        _detected(s, 'neutral')
        problems = cl.preflight(s, 'swing_1', rescale=True, reuse_keypoints=False,
                                keypoints_detected=True)
        assert not any('detected keypoints' in p for p in problems), problems
    finally:
        shutil.rmtree(root)


def test_swings_exclude_calibration_and_neutral():
    root = tempfile.mkdtemp()
    try:
        s = _session(root, trials=('swing_1', 'swing_2', 'neutral', 'calibration'))
        assert ps.swing_trials(s, 'neutral') == ['swing_1', 'swing_2']
        assert not ps.has_scaled_model(s)
        assert ps.has_scaled_model(_session(os.path.join(root, 'b'), scaled=True))
    finally:
        shutil.rmtree(root)


def test_results_are_named_for_the_swing_optimizer():
    root = tempfile.mkdtemp()
    try:
        work = os.path.join(root, 'work_session')
        out = cl.output_dirs(work)
        for d in (os.path.join(out['opensim'], 'Kinematics'), os.path.join(out['opensim'], 'Model'),
                  os.path.join(out['markers'], 'PostAugmentation_v0.3')):
            os.makedirs(d)
        open(os.path.join(out['opensim'], 'Kinematics', 'swing_1_LSTM.mot'), 'w').write('m')
        open(os.path.join(out['markers'], 'PostAugmentation_v0.3', 'swing_1_LSTM.trc'), 'w').write('t')
        open(os.path.join(out['opensim'], 'Model', 'LaiUhlrich2022_scaled.osim'), 'w').write('o')
        files = ps.collect(work, 'swing_1', os.path.join(root, 'results'))
        assert sorted(os.path.basename(p) for p in files.values()) == \
            ['LaiUhlrich2022_scaled.osim', 'swing_1.mot', 'swing_1.trc']
    finally:
        shutil.rmtree(root)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('✓', name)
