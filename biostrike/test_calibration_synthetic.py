"""Calibration round trip: a camera at a known pose films the checkerboard;
recover the pose with OpenCap's own calibration code.

The board is rendered through the deployed iPhone lens model (intrinsics and
distortion), recorded the way a phone browser does (portrait, 30 fps, no
orientation tag), converted by capture_server.media, then handed to the
fork's rotateIntrinsics + calcExtrinsicsFromVideo exactly as main.py does.
Passing means our recordings land in the frame the lens calibration was
measured in, with the right orientation handling, so the solved camera sits
where it really was.

Needs the pipeline's dependencies (OpenCV, ffmpeg/ffprobe). Run:
    python biostrike/test_calibration_synthetic.py
"""
import os
import pickle
import shutil
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, 'capture_server'))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
from capture_local import compat_shims, offline_environment  # noqa: E402
offline_environment(); compat_shims()
import media  # noqa: E402
from utilsChecker import calcExtrinsicsFromVideo, rotateIntrinsics  # noqa: E402

MODEL = 'iPhone10,3'
SQUARE_MM = 35.0
INNER = (5, 4)                     # black-to-black inner corners, as in sessionMetadata
W, H = 720, 1280                   # portrait frame the intrinsics describe


def _pose(yaw_deg, dist_mm):
    """Board-to-camera rotation/translation with the board centred in view."""
    R, _ = cv2.Rodrigues(np.array([0.0, np.radians(yaw_deg), 0.0]))
    centre = np.array([(INNER[0] - 1) * SQUARE_MM / 2, (INNER[1] - 1) * SQUARE_MM / 2, 0.0])
    t = -R @ centre + np.array([0.0, 0.0, dist_mm])
    return R, t


def render_board(K, dist, R, t):
    """Ray-cast every pixel through the lens model onto the board plane."""
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    pix = np.stack([u.ravel(), v.ravel()], axis=1).reshape(-1, 1, 2)
    rays = cv2.undistortPoints(pix, K, dist).reshape(-1, 2)
    d = np.column_stack([rays, np.ones(len(rays))])
    n = R[:, 2]
    s = (n @ t) / (d @ n)
    Xb = (R.T @ (d * s[:, None] - t).T).T            # board-frame mm
    ix, iy = np.floor(Xb[:, 0] / SQUARE_MM), np.floor(Xb[:, 1] / SQUARE_MM)
    on_squares = (ix >= -1) & (ix <= INNER[0] - 1) & (iy >= -1) & (iy <= INNER[1] - 1)
    on_margin = ((Xb[:, 0] >= -2 * SQUARE_MM) & (Xb[:, 0] <= (INNER[0] + 1) * SQUARE_MM) &
                 (Xb[:, 1] >= -2 * SQUARE_MM) & (Xb[:, 1] <= (INNER[1] + 1) * SQUARE_MM))
    img = np.full(len(Xb), 110, np.uint8)            # grey wall
    img[on_margin] = 235                              # white board margin
    img[on_squares & ((ix + iy) % 2 == 0)] = 20       # black squares
    img[on_squares & ((ix + iy) % 2 != 0)] = 235
    return cv2.cvtColor(img.reshape(H, W), cv2.COLOR_GRAY2BGR)


def _camera_centre(R, t):
    return -R.T @ np.asarray(t).ravel()


def run_case(yaw_deg, dist_mm, workdir):
    params = pickle.load(open(os.path.join(REPO, 'CameraIntrinsics', MODEL, 'Deployed',
                                           'cameraIntrinsics.pickle'), 'rb'))
    K, dist = params['intrinsicMat'], params['distortion']
    R, t = _pose(yaw_deg, dist_mm)
    frame = render_board(K, dist, R, t)

    raw = os.path.join(workdir, 'browser.mp4')
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*'mp4v'), 30, (W, H))
    for _ in range(90):
        vw.write(frame)
    vw.release()
    trial_dir = os.path.join(workdir, 'Cam0', 'InputMedia', 'calibration')
    video = os.path.join(trial_dir, 'calibration.mov')
    media.normalize_video(raw, video)

    want = _camera_centre(R, t)
    # A 5x4 board reads the same rotated 180 deg in its plane, which only
    # renumbers the corners; compare against both labellings.
    flip = np.array([(INNER[0] - 1) * SQUARE_MM, (INNER[1] - 1) * SQUARE_MM, 0.0])
    want_flipped = np.array([flip[0] - want[0], flip[1] - want[1], want[2]])
    # A small flat board has two near-equal pose solutions (planar PnP
    # ambiguity: same corners within a fraction of a pixel, camera tilted the
    # other way). OpenCap computes both and picks one later from the
    # athlete's keypoints (autoSelectExtrinsicSolution), so the check is that
    # one of its two candidates is the true camera.
    errs = []
    for second in (False, True):
        cam = rotateIntrinsics(dict(params), video)
        solved = calcExtrinsicsFromVideo(video, cam, {'dimensions': INNER, 'squareSize': SQUARE_MM},
                                         imageUpsampleFactor=4, useSecondExtrinsicsSolution=second)
        got = _camera_centre(solved['rotation'], solved['translation'])
        errs.append(min(np.linalg.norm(got - want), np.linalg.norm(got - want_flipped)))
    return min(errs), np.linalg.norm(want)


def test_calibration_recovers_known_camera_position():
    # Angled views, as in a real setup. Exactly face-on is the degenerate case
    # for a flat board: the two candidates straddle the truth (+/- ~240 mm
    # at 1.5 m), which is why phones should see the board from an angle.
    for yaw, dist in ((20, 1500), (35, 1800), (-30, 1300), (-45, 2000)):
        with tempfile.TemporaryDirectory() as d:
            err, rng = run_case(yaw, dist, d)
        print(f'  yaw {yaw:+3d} deg at {dist} mm: camera position error {err:.1f} mm '
              f'({100 * err / rng:.2f}% of range)')
        assert err / rng < 0.025, f'calibration off by {err:.1f} mm at yaw {yaw}'


if __name__ == '__main__':
    if not shutil.which('ffprobe'):
        sys.exit('needs ffmpeg/ffprobe on PATH')
    test_calibration_recovers_known_camera_position()
    print('✓ test_calibration_recovers_known_camera_position')
