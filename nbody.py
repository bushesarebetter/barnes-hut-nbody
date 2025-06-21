import argparse
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

G = 1.0


def inv_r3(r2, eps2):
    # (r^2 + eps^2)^(-3/2), with self-pairs (r = 0) contributing nothing
    with np.errstate(divide="ignore"):
        out = (r2 + eps2) ** -1.5
    out[r2 == 0] = 0.0
    return out


def direct_accel(pos, mass, eps=0.0, chunk=1024):
    acc = np.zeros_like(pos)
    for s in range(0, len(pos), chunk):
        d = pos[None, :, :] - pos[s:s + chunk, None, :]
        w = mass[None, :] * inv_r3((d ** 2).sum(-1), eps * eps)
        acc[s:s + chunk] = G * (d * w[..., None]).sum(1)
    return acc


class Node:
    __slots__ = ("mass", "com", "size", "children", "bodies")


def build(pos, mass, idx, center, half, depth=0):
    node = Node()
    m = mass[idx]
    node.mass = m.sum()
    node.com = (pos[idx] * m[:, None]).sum(0) / node.mass
    node.size = 2 * half
    node.children = []
    node.bodies = None
    if len(idx) == 1 or depth == 48:  # depth cap only matters for (near-)coincident bodies
        node.bodies = idx
        return node
    east = pos[idx, 0] > center[0]
    north = pos[idx, 1] > center[1]
    for ex in (False, True):
        for no in (False, True):
            sub = idx[(east == ex) & (north == no)]
            if len(sub):
                c = center + 0.5 * half * np.array([1 if ex else -1, 1 if no else -1])
                node.children.append(build(pos, mass, sub, c, 0.5 * half, depth + 1))
    return node


def walk(node, pos, mass, idx, acc, theta, eps2):
    # Walks the tree for a whole batch of target bodies at once: the ones that see this
    # node as far away take its monopole, the rest go one level deeper.
    if node.bodies is not None:
        d = pos[node.bodies][None, :, :] - pos[idx][:, None, :]
        w = mass[node.bodies][None, :] * inv_r3((d ** 2).sum(-1), eps2)
        acc[idx] += G * (d * w[..., None]).sum(1)
        return
    d = node.com - pos[idx]
    r2 = (d ** 2).sum(1)
    far = node.size ** 2 < theta ** 2 * r2
    if far.any():
        acc[idx[far]] += G * node.mass * d[far] * inv_r3(r2[far], eps2)[:, None]
    near = idx[~far]
    if len(near):
        for child in node.children:
            walk(child, pos, mass, near, acc, theta, eps2)


def bh_accel(pos, mass, theta=0.5, eps=0.0):
    lo, hi = pos.min(0), pos.max(0)
    half = 0.5 * (hi - lo).max() * 1.001 + 1e-12
    idx = np.arange(len(pos))
    root = build(pos, mass, idx, 0.5 * (lo + hi), half)
    acc = np.zeros_like(pos)
    walk(root, pos, mass, idx, acc, theta, eps * eps)
    return acc


def leapfrog_step(pos, vel, acc, dt, accel):
    # kick-drift-kick; returns the acceleration at the new positions for the next step
    vel += 0.5 * dt * acc
    pos += dt * vel
    acc = accel(pos)
    vel += 0.5 * dt * acc
    return acc


def energy(pos, vel, mass, eps=0.0):
    kin = 0.5 * (mass * (vel ** 2).sum(1)).sum()
    i, j = np.triu_indices(len(pos), 1)
    r = np.sqrt(((pos[i] - pos[j]) ** 2).sum(1) + eps * eps)
    return kin - G * (mass[i] * mass[j] / r).sum()














