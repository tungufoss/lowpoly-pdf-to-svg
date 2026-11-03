"""Locations behind viewer inspection messages; never changes the paper or folded pose."""
from itertools import combinations
import re
import numpy as np
from scipy.optimize import least_squares


def inspection_targets(model):
    vertices = np.asarray(model['vertices'], dtype=float)
    result = {}
    for piece, data in model['pieces'].items():
        reason = data.get('inspect', '')
        if not reason:
            continue
        targets = []
        faces = [face for face in model['faces'] if face['piece'] == piece]
        flat = {v: point for face in faces for v, point in zip(face['v'], face['flat'])}
        if 'a corner' in reason and flat:
            ids = sorted(flat)
            local = {v: i for i, v in enumerate(ids)}
            pairs = sorted({tuple(sorted(pair)) for face in faces for pair in combinations(face['v'], 2)})
            a = np.array([local[u] for u, v in pairs]); b = np.array([local[v] for u, v in pairs])
            lengths = np.array([np.linalg.norm(vertices[u] - vertices[v]) for u, v in pairs])
            start = np.array([flat[v] for v in ids], dtype=float)
            def residual(q):
                q = q.reshape(-1, 2)
                return np.r_[np.linalg.norm(q[a] - q[b], axis=1) - lengths, 0.05 * (q - start).ravel()]
            fitted = least_squares(residual, start.ravel()).x.reshape(-1, 2)
            delta = fitted - start
            delta -= delta.mean(axis=0)
            boundary = {}
            for face in faces:
                for u, v in zip(face['v'], face['v'][1:] + face['v'][:1]):
                    edge = tuple(sorted((u, v)))
                    boundary[edge] = boundary.get(edge, 0) + 1
            corners = {v for edge, count in boundary.items() if count == 1 for v in edge}
            for i, vertex in enumerate(ids):
                mm = float(np.linalg.norm(delta[i]))
                if vertex in corners and mm > 1.0:
                    targets.append({'kind': 'corner', 'piece': piece, 'vertex': vertex,
                                    'flat': list(flat[vertex]), 'mm': round(mm, 2),
                                    'delta': np.round(delta[i], 3).tolist()})
        held = reason.split('moves held back, they would part glued edges: ', 1)[-1].split(';')[0] if 'moves held back' in reason else ''
        for glue in model['glues']:
            if piece not in (glue['piece'], glue['tab_piece']):
                continue
            if glue['status'] in ('closes', 'proposed, closes') and str(glue['number']) not in re.findall(r'\d+', held):
                continue
            edge = glue['v'][:2] if piece == glue['piece'] else glue['v'][2:]
            if all(v in flat for v in edge):
                targets.append({'kind': 'edge', 'piece': piece, 'vertices': edge,
                                'flat': [flat[v] for v in edge], 'number': str(glue['number']),
                                'mm': float(glue['gap_mm'])})
        result[piece] = targets
    return result
