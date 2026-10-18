"""Export a swing as real OpenSim bones for the Swing Optimizer's 3-D view.

    python biostrike/export_bones.py --model <scaled.osim> --mot <ik.mot> \
        --geometry <OpenSimData/Model/Geometry> --out <folder> [--start 3.4 --end 4.4]

Writes two files:
  bones_motion.json    every body's position and rotation in ground for each
                       frame, from OpenCap's own generateVisualizerJson (the data
                       OpenCap's web viewer animates), trimmed to the window
  bones_geometry.json  the bone meshes those bodies carry, converted from the
                       session's .vtp files to compact triangle arrays

The viewer places each mesh at its body's transform (body-fixed XYZ rotation),
scaled by the body's scale factors, which is how OpenCap's viewer
(opencap-org/opencap-viewer, Apache 2.0) renders a trial.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def read_vtp(path: str) -> tuple[np.ndarray, np.ndarray]:
    """(points float32 Nx3, triangles uint32 Mx3) from an ASCII VTK PolyData file."""
    root = ET.parse(path).getroot()
    piece = root.find('.//Piece')

    def array(parent: str, name: str | None = None) -> np.ndarray:
        node = piece.find(parent)
        for da in node.findall('DataArray'):
            if name is None or da.get('Name') == name:
                if da.get('format') != 'ascii':
                    raise ValueError(f'{path}: only ascii DataArrays are supported')
                return np.array(da.text.split(), dtype=float)
        raise ValueError(f'{path}: no {parent}/{name}')

    pts = array('Points').reshape(-1, 3).astype(np.float32)
    conn = array('Polys', 'connectivity').astype(np.int64)
    offs = array('Polys', 'offsets').astype(np.int64)
    tris, start = [], 0
    for end in offs:                     # fan-triangulate each polygon
        poly = conn[start:end]
        for k in range(1, len(poly) - 1):
            tris.append((poly[0], poly[k], poly[k + 1]))
        start = end
    return pts, np.array(tris, dtype=np.uint32)


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode('ascii')


def pack_geometry(names: list[str], geometry_dir: str) -> dict:
    out = {}
    for name in sorted(set(names)):
        path = os.path.join(geometry_dir, name)
        if not os.path.exists(path):
            print('missing mesh, skipped:', name)
            continue
        pts, tris = read_vtp(path)
        idx = tris.astype(np.uint16 if len(pts) < 65536 else np.uint32)
        out[name] = {'positions': b64(pts), 'indices': b64(idx),
                     'index_type': 'u16' if idx.dtype == np.uint16 else 'u32'}
    return out


def motion(model: str, mot: str, start: float | None, end: float | None) -> dict:
    sys.path.insert(0, HERE)
    from capture_local import compat_shims, offline_environment
    offline_environment(); compat_shims()
    sys.path.insert(0, REPO)
    from utilsOpenSim import generateVisualizerJson
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'vis.json')
        generateVisualizerJson(model, mot, path, roundToRotations=4, roundToTranslations=4)
        vis = json.load(open(path))
    t = np.asarray(vis['time'])
    keep = np.ones(len(t), bool)
    if start is not None:
        keep &= t >= start - 1e-9
    if end is not None:
        keep &= t <= end + 1e-9
    idx = np.flatnonzero(keep)
    bodies = {}
    for name, b in vis['bodies'].items():
        if not b['attachedGeometries']:
            continue
        bodies[name] = {'geometries': b['attachedGeometries'], 'scale': b['scaleFactors'],
                        'rotation': [b['rotation'][i] for i in idx],
                        'translation': [b['translation'][i] for i in idx]}
    times = t[idx]
    return {'format': 'biostrike.bones.v1', 'time': [round(float(x), 4) for x in times],
            'fps': round(float(1 / np.median(np.diff(times))), 2) if len(times) > 1 else None,
            'bodies': bodies}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--model', required=True)
    ap.add_argument('--mot', required=True)
    ap.add_argument('--geometry', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--start', type=float)
    ap.add_argument('--end', type=float)
    ap.add_argument('--contact', type=float, help='time of contact, stored for the viewer')
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    m = motion(args.model, args.mot, args.start, args.end)
    if args.contact is not None and m['time']:
        m['contact_frame'] = int(np.argmin(np.abs(np.asarray(m['time']) - args.contact)))
    geoms = [g for b in m['bodies'].values() for g in b['geometries']]
    g = pack_geometry(geoms, args.geometry)
    with open(os.path.join(args.out, 'bones_motion.json'), 'w') as f:
        json.dump(m, f, separators=(',', ':'))
    with open(os.path.join(args.out, 'bones_geometry.json'), 'w') as f:
        json.dump({'format': 'biostrike.bone-meshes.v1', 'meshes': g}, f, separators=(',', ':'))
    for name in ('bones_motion.json', 'bones_geometry.json'):
        print(f'{name}: {os.path.getsize(os.path.join(args.out, name)) / 1024:.0f} KB')
    print(f"{len(m['bodies'])} bodies, {len(g)} meshes, {len(m['time'])} frames")
    return 0


if __name__ == '__main__':
    sys.exit(main())
