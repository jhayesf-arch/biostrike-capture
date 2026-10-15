"""Run the OpenCap pipeline locally on a downloaded OpenCap session.

    python biostrike/capture_local.py --session /path/to/OpenCapData_<id> \
        --trial Quinn_1_60fps_1 --workdir ./runs [--reuse-opencap-keypoints] [--rescale]

Takes a session as downloaded from app.opencap.ai (Videos/Cam*, sessionMetadata,
calibration pickles, OpenSimData) and runs this fork's main.py on it, with no
calls to Stanford's servers. The upstream worker (app.py) polls Stanford's queue;
this replaces it for local and self-hosted use.

Choices this wrapper enforces:

  MMPose only.     OpenPose is licensed for non-commercial use only. Upstream's
                   default is OpenPose; this runs MMPose (HRNet) — which is also
                   what OpenCap's cloud uses (posemodel: hrnet in the session
                   metadata), so results are comparable.

  Never in place.  main.py writes into the session folder. The session is copied
                   into --workdir, and outputs go to a separate
                   MarkerData/mmpose_0.8/biostrike and OpenSimData/mmpose_0.8/biostrike
                   tree, so OpenCap's own .trc/.mot in the copy stay intact as the
                   reference for compare_to_opencap.py.

  Calibration is reused when the session has it (cameraIntrinsicsExtrinsics.pickle
  per camera, as in an OpenCap download). Sessions recorded with our own capture
  server have a "calibration" checkerboard trial instead; it is solved first,
  with OpenCap's own extrinsics code, before any other trial.

--reuse-opencap-keypoints seeds the pose-detector cache with the keypoints OpenCap
already computed (Videos/Cam*/OutputPkl/<trial>_keypoints.pkl). main.py skips
pose detection when that cache exists, so the run needs no GPU — it exercises
sync, triangulation, marker augmentation and OpenSim IK only. Without the flag,
pose detection runs and needs MMPose installed, realistically on an NVIDIA GPU.

--rescale re-runs model scaling from the session's neutral trial; by default the
scaled model from the download is reused, so a mismatch can only come from the
trial itself.

Needs: this fork's Python dependencies, OpenSim (conda), and ffmpeg on PATH
(main.py writes a rotated copy of each video even when pose detection is skipped).
"""

from __future__ import annotations

import argparse
import glob
import os
import pickletools
import shutil
import sys
import time
from typing import List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

BBOX_THR = 0.8                       # main.py's default; also names the cache folder
OUTPUT_SUFFIX = 'biostrike'          # markerDataFolderNameSuffix for our outputs
POSE_DETECTOR = 'mmpose'
VIDEO_EXTS = ('.mov', '.mp4', '.avi')
CALIBRATION_TRIAL = 'calibration'


# ── Session preparation (pure file work, no pipeline imports) ──────────────

def camera_dirs(session_dir: str) -> List[str]:
    return sorted(d for d in glob.glob(os.path.join(session_dir, 'Videos', 'Cam*'))
                  if os.path.isdir(d))


def trial_video(cam_dir: str, trial: str) -> Optional[str]:
    for ext in VIDEO_EXTS:
        p = os.path.join(cam_dir, 'InputMedia', trial, trial + ext)
        if os.path.exists(p):
            return p
    return None


def needs_calibration(session_dir: str) -> bool:
    return any(not os.path.exists(os.path.join(cam, 'cameraIntrinsicsExtrinsics.pickle'))
               for cam in camera_dirs(session_dir))


def output_dirs(session_dir: str) -> dict:
    sub = os.path.join(f'{POSE_DETECTOR}_{BBOX_THR}', OUTPUT_SUFFIX)
    return {'markers': os.path.join(session_dir, 'MarkerData', sub),
            'opensim': os.path.join(session_dir, 'OpenSimData', sub)}


def preflight(session_dir: str, trial: str, rescale: bool, reuse_keypoints: bool,
              neutral: str = 'neutral') -> List[str]:
    """Everything that would make main.py fail, as readable problems. [] = ready."""
    problems = []
    if not os.path.exists(os.path.join(session_dir, 'sessionMetadata.yaml')):
        problems.append('sessionMetadata.yaml is missing from the session root.')
    cams = camera_dirs(session_dir)
    if len(cams) < 2:
        problems.append(f'Need at least 2 camera folders under Videos/, found {len(cams)}. '
                        'Upload the whole session folder so Cam0/Cam1 stay separate — '
                        'files with identical names from different cameras cannot be '
                        'told apart once pulled out.')
    for cam in cams:
        name = os.path.basename(cam)
        if not trial_video(cam, trial):
            problems.append(f'{name}: no video at InputMedia/{trial}/{trial}.mov|.mp4')
        if not (os.path.exists(os.path.join(cam, 'cameraIntrinsicsExtrinsics.pickle'))
                or trial_video(cam, CALIBRATION_TRIAL)):
            problems.append(f'{name}: no calibration — neither cameraIntrinsicsExtrinsics.pickle '
                            f'nor a "{CALIBRATION_TRIAL}" checkerboard video.')
        if reuse_keypoints and not os.path.exists(
                os.path.join(cam, 'OutputPkl', f'{trial}_keypoints.pkl')):
            problems.append(f'{name}: OutputPkl/{trial}_keypoints.pkl is missing, '
                            'so pose detection cannot be skipped.')
    if rescale:
        for cam in cams:
            if not trial_video(cam, neutral):
                problems.append(f'{os.path.basename(cam)}: --rescale needs the '
                                f'"{neutral}" trial video.')
    elif not glob.glob(os.path.join(session_dir, 'OpenSimData', 'Model', '*_scaled.osim')):
        problems.append('No scaled model in OpenSimData/Model — pass --rescale to build '
                        'one from the neutral trial.')
    if shutil.which('ffmpeg') is None:
        problems.append('ffmpeg is not on PATH (main.py uses it to rotate videos).')
    if needs_calibration(session_dir) and shutil.which('ffprobe') is None:
        problems.append('ffprobe is not on PATH (calibration reads video length with it).')
    return problems


def copy_session(src: str, workdir: str, fresh: bool = False) -> str:
    """Copy the download into <workdir>/Data/<name>; main.py reads from there."""
    name = os.path.basename(os.path.normpath(src))
    dst = os.path.join(workdir, 'Data', name)
    if os.path.exists(dst) and fresh:
        shutil.rmtree(dst)
    if not os.path.exists(dst):
        shutil.copytree(src, dst)
    return dst


def _is_plain_data_pickle(path: str) -> bool:
    """True if loading the pickle cannot import or call anything.

    main.py unpickles these files. A keypoint cache is plain lists, dicts and
    numbers; refusing anything with GLOBAL/REDUCE opcodes means a doctored file
    can't run code when it is loaded.
    """
    with open(path, 'rb') as f:
        return not any(op.name in ('GLOBAL', 'STACK_GLOBAL', 'INST', 'OBJ', 'REDUCE',
                                   'NEWOBJ', 'NEWOBJ_EX', 'BUILD')
                       for op, _, _ in pickletools.genops(f))


def seed_keypoint_cache(session_dir: str, trial: str) -> List[str]:
    """Place OpenCap's keypoints where main.py's MMPose step looks for its cache.

    runMMposeVideo skips detection when
    <Cam>/OutputPkl_mmpose_<bbox>/<trial>/<trial>_rotated_pp.pkl exists.
    The download's OutputPkl/<trial>_keypoints.pkl is that same post-processed
    file: per frame, a list of people, each with 25 keypoints × (x, y, confidence).
    """
    placed = []
    for cam in camera_dirs(session_dir):
        src = os.path.join(cam, 'OutputPkl', f'{trial}_keypoints.pkl')
        if not _is_plain_data_pickle(src):
            raise ValueError(f'{src} contains executable pickle opcodes; refusing to use it.')
        dst_dir = os.path.join(cam, f'OutputPkl_{POSE_DETECTOR}_{BBOX_THR}', trial)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, f'{trial}_rotated_pp.pkl')
        shutil.copy2(src, dst)
        placed.append(dst)
    return placed


def seed_scaled_model(session_dir: str) -> str:
    """Reuse the downloaded scaled model for our output tree (no --rescale)."""
    src = sorted(glob.glob(os.path.join(session_dir, 'OpenSimData', 'Model', '*_scaled.osim')))[0]
    dst_dir = os.path.join(output_dirs(session_dir)['opensim'], 'Model')
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    shutil.copy2(src, dst)
    return dst


def find_outputs(session_dir: str, trial: str, since: float) -> dict:
    out = output_dirs(session_dir)
    def newest(pattern):
        hits = [p for p in glob.glob(pattern, recursive=True) if os.path.getmtime(p) >= since]
        return max(hits, key=os.path.getmtime) if hits else None
    return {'trc': newest(os.path.join(out['markers'], 'PostAugmentation*', f'{trial}*.trc')),
            'mot': newest(os.path.join(out['opensim'], 'Kinematics', f'{trial}*.mot'))}


def find_references(session_dir: str, trial: str) -> dict:
    trc = os.path.join(session_dir, 'MarkerData', f'{trial}.trc')
    mot = os.path.join(session_dir, 'OpenSimData', 'Kinematics', f'{trial}.mot')
    return {'trc': trc if os.path.exists(trc) else None,
            'mot': mot if os.path.exists(mot) else None}


# ── Pipeline call ──────────────────────────────────────────────────────────

def offline_environment() -> None:
    """Make importing upstream code safe without a Stanford account.

    utils.py and utilsChecker.py call getToken() at IMPORT time, which prompts
    for an app.opencap.ai username and password and posts them to
    api.opencap.ai — before any local processing starts. getToken() returns
    early when API_TOKEN is set, so a placeholder skips the prompt. The token
    is only used by requests to Stanford's API (calibration's
    computeAverageIntrinsics), which this runner never reaches because it
    reuses the session's calibration.

    API_URL is pointed at a reserved, unresolvable name, so if a future
    upstream change does make a request on this path it fails at once rather
    than silently reaching Stanford's servers.
    """
    os.environ.setdefault('API_TOKEN', 'offline-biostrike')
    os.environ.setdefault('API_URL', 'http://opencap-api.invalid/')


def compat_shims() -> None:
    """Bridge library APIs that moved after upstream pinned its versions.

    Upstream targets SciPy < 1.13 and NumPy 1.x. OpenSim's pip wheel requires
    NumPy >= 2.1, which in turn requires SciPy >= 1.13, where
    scipy.signal.gaussian was removed in favour of scipy.signal.windows.gaussian
    (same function). Aliasing it here keeps upstream files unmodified.
    """
    import scipy.signal
    if not hasattr(scipy.signal, 'gaussian'):
        from scipy.signal.windows import gaussian
        scipy.signal.gaussian = gaussian


def run_pipeline(workdir: str, session_name: str, trial: str, scale: bool,
                 extrinsics: bool = False) -> None:
    """Call upstream main.main() — imported lazily, since it pulls in TensorFlow,
    OpenSim and OpenCV, none of which the preparation steps need."""
    offline_environment()
    compat_shims()
    # Upstream runs OpenSim tools by shelling out to `opensim-cmd`, which the
    # pip wheel does not ship. Fall back to biostrike/bin/opensim-cmd (same
    # setup file, run through the Python API), with this interpreter first on
    # PATH so the shim imports the same opensim.
    if shutil.which('opensim-cmd') is None:
        os.environ['PATH'] = os.pathsep.join(
            [os.path.join(HERE, 'bin'), os.path.dirname(sys.executable), os.environ.get('PATH', '')])
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from main import main as opencap_main
    opencap_main(session_name, trial, trial,
                 dataDir=os.path.abspath(workdir),
                 poseDetector=POSE_DETECTOR, bbox_thr=BBOX_THR,
                 isDocker=False, scaleModel=scale, extrinsicsTrial=extrinsics,
                 genericFolderNames=False, markerDataFolderNameSuffix=OUTPUT_SUFFIX)


# ── CLI ────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--session', required=True, help='downloaded OpenCap session folder')
    ap.add_argument('--trial', required=True, help='trial name, e.g. Quinn_1_60fps_1')
    ap.add_argument('--workdir', default='runs', help='where the working copy goes')
    ap.add_argument('--reuse-opencap-keypoints', action='store_true',
                    help="seed the pose cache with OpenCap's keypoints: no GPU needed")
    ap.add_argument('--rescale', action='store_true',
                    help='re-run model scaling from the neutral trial')
    ap.add_argument('--neutral', default='neutral', help='name of the neutral trial')
    ap.add_argument('--fresh', action='store_true', help='discard an existing working copy')
    ap.add_argument('--check-only', action='store_true', help='preflight only, do not run')
    args = ap.parse_args(argv)

    problems = preflight(args.session, args.trial, args.rescale,
                         args.reuse_opencap_keypoints, args.neutral)
    if problems:
        print('Not ready to run:')
        for p in problems:
            print('  -', p)
        return 2
    if args.check_only:
        print('Preflight passed.')
        return 0

    session_dir = copy_session(args.session, args.workdir, args.fresh)
    session_name = os.path.basename(session_dir)
    if args.reuse_opencap_keypoints:
        for p in seed_keypoint_cache(session_dir, args.trial):
            print('seeded keypoints:', os.path.relpath(p, session_dir))
    start = time.time()
    if needs_calibration(session_dir):
        print('calibrating cameras from the checkerboard trial…')
        run_pipeline(args.workdir, session_name, CALIBRATION_TRIAL, scale=False, extrinsics=True)
        missing = [os.path.basename(c) for c in camera_dirs(session_dir)
                   if not os.path.exists(os.path.join(c, 'cameraIntrinsicsExtrinsics.pickle'))]
        if missing:
            print('Calibration failed for', ', '.join(missing))
            return 3
    if args.rescale:
        if args.reuse_opencap_keypoints:
            seed_keypoint_cache(session_dir, args.neutral)
        run_pipeline(args.workdir, session_name, args.neutral, scale=True)
    else:
        print('reusing scaled model:', os.path.relpath(seed_scaled_model(session_dir), session_dir))
    run_pipeline(args.workdir, session_name, args.trial, scale=False)

    outs, refs = find_outputs(session_dir, args.trial, start), find_references(session_dir, args.trial)
    print('\noutputs:', outs)
    status = 0
    from compare_to_opencap import compare_mot, compare_trc, _print_report
    for kind, fn in (('trc', compare_trc), ('mot', compare_mot)):
        if outs[kind] and refs[kind]:
            report = fn(outs[kind], refs[kind])
            _print_report(report)
            status = status or (0 if report['passed'] else 1)
        else:
            print(f'{kind}: nothing to compare (ours={outs[kind]}, reference={refs[kind]})')
    return status


if __name__ == '__main__':
    sys.path.insert(0, HERE)
    sys.exit(main())
