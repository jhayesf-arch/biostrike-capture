"""Tests for capture_local's session handling. No TensorFlow/OpenSim needed:
the upstream pipeline call is replaced with a stand-in that writes outputs."""
import os
import pickle
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capture_local as cl  # noqa: E402

TRIAL = 'Quinn_1_60fps_1'
TRC = ('PathFileType\t4\t(X/Y/Z)\tx\n'
       'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n'
       '60\t60\t3\t1\tm\t60\t1\t3\n'
       'Frame#\tTime\tNeck\t\t\n\t\tX1\tY1\tZ1\n'
       '1\t0.000\t0.1\t1.5\t0.2\n2\t0.0167\t0.1\t1.5\t0.2\n3\t0.0333\t0.1\t1.5\t0.2\n')
MOT = ('Coordinates\nversion=1\nnRows=3\nnColumns=2\ninDegrees=yes\nendheader\n'
       'time\tknee_angle_r\n0.000\t10\n0.0167\t11\n0.0333\t12\n')


def _fake_session(root, n_cams=2, keypoints_obj=None):
    s = os.path.join(root, 'OpenCapData_test')
    open(os.path.join(os.makedirs(s) or s, 'sessionMetadata.yaml'), 'w').write('openSimModel: LaiUhlrich2022\n')
    for i in range(n_cams):
        cam = os.path.join(s, 'Videos', f'Cam{i}')
        os.makedirs(os.path.join(cam, 'InputMedia', TRIAL))
        open(os.path.join(cam, 'InputMedia', TRIAL, TRIAL + '.mov'), 'wb').write(b'video')
        open(os.path.join(cam, 'cameraIntrinsicsExtrinsics.pickle'), 'wb').write(b'cal')
        os.makedirs(os.path.join(cam, 'OutputPkl'))
        with open(os.path.join(cam, 'OutputPkl', f'{TRIAL}_keypoints.pkl'), 'wb') as f:
            pickle.dump(keypoints_obj if keypoints_obj is not None else
                        [[{'person_id': [0], 'pose_keypoints_2d': [0.0] * 75}]], f)
    os.makedirs(os.path.join(s, 'OpenSimData', 'Model'))
    open(os.path.join(s, 'OpenSimData', 'Model', 'LaiUhlrich2022_scaled.osim'), 'w').write('<m/>')
    os.makedirs(os.path.join(s, 'OpenSimData', 'Kinematics'))
    open(os.path.join(s, 'OpenSimData', 'Kinematics', TRIAL + '.mot'), 'w').write(MOT)
    os.makedirs(os.path.join(s, 'MarkerData'))
    open(os.path.join(s, 'MarkerData', TRIAL + '.trc'), 'w').write(TRC)
    return s


def _with_ffmpeg(fn):
    def wrapped():
        orig = shutil.which
        shutil.which = lambda name: '/usr/bin/ffmpeg' if name == 'ffmpeg' else orig(name)
        try:
            fn()
        finally:
            shutil.which = orig
    wrapped.__name__ = fn.__name__
    return wrapped


@_with_ffmpeg
def test_preflight_ready_session_passes():
    with tempfile.TemporaryDirectory() as d:
        assert cl.preflight(_fake_session(d), TRIAL, rescale=False, reuse_keypoints=True) == []


@_with_ffmpeg
def test_preflight_explains_single_camera():
    with tempfile.TemporaryDirectory() as d:
        problems = cl.preflight(_fake_session(d, n_cams=1), TRIAL, False, True)
    assert any('at least 2 camera' in p for p in problems)


def test_seed_keypoints_lands_where_main_looks():
    with tempfile.TemporaryDirectory() as d:
        s = _fake_session(d)
        placed = cl.seed_keypoint_cache(s, TRIAL)
        expect = os.path.join(s, 'Videos', 'Cam0', 'OutputPkl_mmpose_0.8', TRIAL,
                              f'{TRIAL}_rotated_pp.pkl')
        assert expect in placed and os.path.exists(expect)


def test_seed_keypoints_refuses_code_carrying_pickle():
    class Evil:                       # pickles to a REDUCE call on load
        def __reduce__(self):
            return (print, ('would run',))
    with tempfile.TemporaryDirectory() as d:
        s = _fake_session(d, keypoints_obj=[Evil()])
        try:
            cl.seed_keypoint_cache(s, TRIAL)
        except ValueError as e:
            assert 'executable' in str(e)
        else:
            raise AssertionError('accepted a pickle that executes code on load')


@_with_ffmpeg
def test_full_run_keeps_reference_and_compares():
    """End to end with the pipeline stubbed: the source download is untouched,
    outputs land in the separate biostrike tree, and the comparison runs."""
    calls = []

    def fake_pipeline(workdir, session_name, trial, scale):
        calls.append((trial, scale))
        sdir = os.path.join(workdir, 'Data', session_name)
        out = cl.output_dirs(sdir)
        os.makedirs(os.path.join(out['markers'], 'PostAugmentation_v0.3'), exist_ok=True)
        open(os.path.join(out['markers'], 'PostAugmentation_v0.3', trial + '.trc'), 'w').write(TRC)
        os.makedirs(os.path.join(out['opensim'], 'Kinematics'), exist_ok=True)
        open(os.path.join(out['opensim'], 'Kinematics', trial + '.mot'), 'w').write(MOT)

    orig = cl.run_pipeline
    cl.run_pipeline = fake_pipeline
    try:
        with tempfile.TemporaryDirectory() as d:
            src = _fake_session(d)
            before = sorted(os.path.relpath(os.path.join(r, f), src)
                            for r, _, fs in os.walk(src) for f in fs)
            work = os.path.join(d, 'runs')
            status = cl.main(['--session', src, '--trial', TRIAL, '--workdir', work,
                              '--reuse-opencap-keypoints'])
            after = sorted(os.path.relpath(os.path.join(r, f), src)
                           for r, _, fs in os.walk(src) for f in fs)
            assert before == after, 'the downloaded session must not be modified'
            copy = os.path.join(work, 'Data', 'OpenCapData_test')
            assert open(os.path.join(copy, 'MarkerData', TRIAL + '.trc')).read() == TRC
            assert os.path.exists(os.path.join(copy, 'OpenSimData', 'mmpose_0.8', 'biostrike',
                                               'Model', 'LaiUhlrich2022_scaled.osim'))
    finally:
        cl.run_pipeline = orig
    assert calls == [(TRIAL, False)]
    assert status == 0


if __name__ == '__main__':
    tests = [v for k, v in dict(globals()).items() if k.startswith('test_')]
    for fn in tests:
        fn(); print(f'✓ {fn.__name__}')
    print(f'{len(tests)} passed')
