"""Download a session processed on app.opencap.ai into Swing Optimizer uploads.

    python biostrike/opencap_download.py <session id or link> --results <folder> \
        [--trials swing_1 swing_2]

Login: set OPENCAP_TOKEN, or you are asked for your app.opencap.ai username and
password (they go only to api.opencap.ai, which returns a token; neither is
saved). For each swing it leaves, in <results>/<trial>/, the same files
process_session.py does:
    <trial>.mot      joint angles (OpenCap's inverse kinematics)
    <trial>.trc      markers (OpenCap's marker-enhanced set)
    <model>.osim     the athlete's scaled model, under OpenCap's own file name
                     (e.g. LaiUhlrich2022_scaled.osim, or ..._shoulder_scaled.osim)
Files already there are skipped, so a cut-off run resumes.

Uses the same public API endpoints as opencap-processing's utils.py
(github.com/opencap-org/opencap-processing, Apache 2.0) with only the standard
library, so it runs in plain Colab without OpenSim or its install time.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Dict, List, Optional

API_URL = os.environ.get('OPENCAP_API_URL', 'https://api.opencap.ai/').rstrip('/') + '/'
NOT_SWINGS = ('calibration', 'neutral')
_UUID = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.I)


# ── HTTP (replaceable in tests) ──────────────────────────────────────────────

def _request(url: str, token: Optional[str] = None, data: Optional[dict] = None) -> bytes:
    headers = {'Authorization': f'Token {token}'} if token else {}
    body = urllib.parse.urlencode(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def session_id_from(text: str) -> str:
    """Accept a bare id or any app.opencap.ai link that contains one."""
    m = _UUID.search(text or '')
    if not m:
        raise ValueError(f'No OpenCap session id in {text!r}. Copy it from the session link.')
    return m.group(0).lower()


def login(username: str, password: str, fetch: Callable = _request) -> str:
    try:
        resp = json.loads(fetch(API_URL + 'login/', data={'username': username, 'password': password}))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'OpenCap login failed (HTTP {e.code}). Check the username and '
                           'password you use at app.opencap.ai.') from None
    if 'token' not in resp:
        raise RuntimeError('OpenCap login did not return a token: ' +
                           ', '.join(f'{k}' for k in resp) +
                           '. If your account uses two-factor login, create a token instead '
                           '(see opencap-processing createAuthenticationEnvFile.py) and set OPENCAP_TOKEN.')
    return resp['token']


def get_json(path: str, token: str, fetch: Callable = _request) -> dict:
    try:
        return json.loads(fetch(API_URL + path, token=token))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise RuntimeError('OpenCap refused the request: the token is invalid or this '
                               'session belongs to another account and is not public.') from None
        if e.code in (404, 500):
            raise RuntimeError(f'OpenCap has no {path!r}. Check the session id.') from None
        raise


# ── What to download ─────────────────────────────────────────────────────────

def result_url(trial: dict, tag: str) -> Optional[str]:
    for r in trial.get('results') or []:
        if r.get('tag') == tag and r.get('media'):
            return r['media']
    return None


def model_file_name(url: str) -> str:
    """OpenCap stores the model as <prefix>-<name>.osim?<signature>."""
    path = urllib.parse.urlparse(url).path
    name = os.path.basename(path)
    return name[name.rfind('-') + 1:] if '-' in name else name


def swing_trials(session: dict, wanted: Optional[List[str]] = None) -> List[dict]:
    trials = sorted(session.get('trials') or [], key=lambda t: t.get('created_at') or '')
    swings = [t for t in trials if t.get('name') not in NOT_SWINGS]
    if wanted:
        names = {t['name'] for t in swings}
        missing = [w for w in wanted if w not in names]
        if missing:
            print('Not in this session:', ', '.join(missing), '| it has:', ', '.join(sorted(names)))
        swings = [t for t in swings if t['name'] in wanted]
    return swings


def neutral_trial_id(session: dict) -> Optional[str]:
    ids = [t['id'] for t in session.get('trials') or [] if t.get('name') == 'neutral']
    if ids:
        return ids[-1]
    meta = (session.get('meta') or {}).get('neutral_trial') or {}
    return meta.get('id')


# ── Download ─────────────────────────────────────────────────────────────────

def _save(url: str, path: str, fetch: Callable) -> bool:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return False
    data = fetch(url)          # signed storage link: no token
    tmp = path + '.part'
    with open(tmp, 'wb') as f:
        f.write(data)
    os.replace(tmp, path)
    return True


def download(session_ref: str, results: str, token: str,
             trials: Optional[List[str]] = None, fetch: Callable = _request) -> Dict[str, dict]:
    sid = session_id_from(session_ref)
    session = get_json(f'sessions/{sid}/', token, fetch)
    if 'trials' not in session:
        raise RuntimeError('This session is not in your account and is not public.')
    os.makedirs(results, exist_ok=True)
    mono = bool(session.get('isMono'))

    model_url = None
    if not mono:
        nid = neutral_trial_id(session)
        if nid:
            neutral = get_json(f'trials/{nid}/', token, fetch)
            model_url = result_url(neutral, 'opensim_model')
            meta = result_url(neutral, 'session_metadata')
            if meta:
                _save(meta, os.path.join(results, 'sessionMetadata.yaml'), fetch)
        if not model_url:
            print('Warning: no scaled model on the neutral trial; swings will lack the .osim.')

    report = {}
    for t in swing_trials(session, trials):
        name = t['name']
        trial = get_json(f"trials/{t['id']}/", token, fetch)
        dest = os.path.join(results, name)
        os.makedirs(dest, exist_ok=True)
        got, missing = [], []
        murl = result_url(trial, 'opensim_model') if mono else model_url
        for tag, fname in (('ik_results', f'{name}.mot'), ('marker_data', f'{name}.trc'),
                           ('opensim_model', model_file_name(murl) if murl else None)):
            url = murl if tag == 'opensim_model' else result_url(trial, tag)
            if not url or not fname:
                missing.append(tag)
                continue
            _save(url, os.path.join(dest, fname), fetch)
            got.append(fname)
        report[name] = {'files': got, 'missing': missing, 'status': trial.get('status')}
        line = f"{name:24s} {', '.join(got) or '-'}"
        if missing:
            line += f"   missing: {', '.join(missing)} (status: {trial.get('status')}; not processed yet?)"
        print(line)
    if not report:
        print('No swing trials found in this session.')
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('session', help='OpenCap session id, or its app.opencap.ai link')
    ap.add_argument('--results', required=True, help='where the upload-ready folders go')
    ap.add_argument('--trials', nargs='*', help='only these trial names')
    args = ap.parse_args(argv)
    token = os.environ.get('OPENCAP_TOKEN')
    if not token:
        print('Log in with the account you use at app.opencap.ai.')
        token = login(input('Username: '), getpass.getpass('Password: '))
    report = download(args.session, args.results, token, args.trials)
    return 0 if report and all(not r['missing'] for r in report.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
