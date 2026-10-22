"""Turn a capture session with detected keypoints into Swing Optimizer uploads.

    python biostrike/process_session.py --session <session folder> \
        --results <folder> [--workdir runs] [--trials swing_1 swing_2]

Runs after detect_session.py (the GPU step) and does everything else: camera
calibration from the checkerboard trial, model scaling from the neutral trial
when the session has no scaled model yet, then triangulation, the marker
enhancer and OpenSim inverse kinematics for every swing. None of it needs a GPU.

For each swing it leaves, in <results>/<trial>/, the three files the Swing
Optimizer takes:
    <trial>.mot                    joint angles
    <trial>.trc                    markers
    LaiUhlrich2022_scaled.osim     the athlete's scaled model
A trial that fails is reported and skipped; the others still run.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
from typing import List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import capture_local as cl            # noqa: E402
from detect_session import trials_in  # noqa: E402


def swing_trials(session_dir: str, neutral: str) -> List[str]:
    return [t for t in trials_in(session_dir) if t not in (cl.CALIBRATION_TRIAL, neutral)]


def has_scaled_model(session_dir: str) -> bool:
    return bool(glob.glob(os.path.join(session_dir, 'OpenSimData', 'Model', '*_scaled.osim')))


def newest(pattern: str) -> Optional[str]:
    hits = glob.glob(pattern)
    return max(hits, key=os.path.getmtime) if hits else None


def collect(work_session: str, trial: str, results: str) -> dict:
    """Copy one trial's .mot, .trc and scaled model into <results>/<trial>/."""
    out = cl.output_dirs(work_session)
    files = {
        'mot': newest(os.path.join(out['opensim'], 'Kinematics', f'{trial}*.mot')),
        'trc': newest(os.path.join(out['markers'], 'PostAugmentation*', f'{trial}*.trc')),
        'osim': (newest(os.path.join(out['opensim'], 'Model', '*_scaled.osim'))
                 or newest(os.path.join(work_session, 'OpenSimData', 'Model', '*_scaled.osim'))),
    }
    dest = os.path.join(results, trial)
    os.makedirs(dest, exist_ok=True)
    names = {'mot': f'{trial}.mot', 'trc': f'{trial}.trc', 'osim': 'LaiUhlrich2022_scaled.osim'}
    copied = {}
    for kind, src in files.items():
        if src:
            copied[kind] = shutil.copy2(src, os.path.join(dest, names[kind]))
    return copied


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--session', required=True)
    ap.add_argument('--results', required=True, help='where the upload-ready files go')
    ap.add_argument('--workdir', default='runs', help='scratch copy of the session')
    ap.add_argument('--trials', nargs='*', help='default: every trial except calibration and neutral')
    ap.add_argument('--neutral', default='neutral')
    args = ap.parse_args(argv)

    session = os.path.abspath(args.session)
    trials = args.trials or swing_trials(session, args.neutral)
    if not trials:
        print('No swing trials found (every camera needs InputMedia/<trial>/<trial>.mov).')
        return 2
    rescale = not has_scaled_model(session)
    if rescale:
        print(f'No scaled model yet: it will be built from the "{args.neutral}" trial.')

    summary = []
    for i, trial in enumerate(trials):
        print(f'\n=== {trial} ({i + 1}/{len(trials)}) ===', flush=True)
        cmd = ['--session', session, '--trial', trial, '--workdir', args.workdir,
               '--keypoints-detected', '--neutral', args.neutral]
        if rescale:
            cmd.append('--rescale')
        try:
            status = cl.main(cmd)
        except Exception as e:      # one bad trial must not stop the rest
            print(f'{trial} failed: {type(e).__name__}: {str(e)[:300]}')
            summary.append((trial, 'failed', {}))
            continue
        if status not in (0, 1):    # 1 = differs from an OpenCap reference; still a result
            summary.append((trial, f'not ready (code {status})', {}))
            continue
        work_session = os.path.join(args.workdir, 'Data', os.path.basename(os.path.normpath(session)))
        files = collect(work_session, trial, args.results)
        summary.append((trial, 'ok' if len(files) == 3 else 'incomplete', files))

    print('\nResults:')
    for trial, state, files in summary:
        print(f'  {trial:24s} {state:12s} {", ".join(sorted(files)) or "-"}')
    print(f'\nUpload each folder in {os.path.abspath(args.results)} to the Swing Optimizer.')
    return 0 if all(s == 'ok' for _, s, _ in summary) else 1


if __name__ == '__main__':
    sys.exit(main())
