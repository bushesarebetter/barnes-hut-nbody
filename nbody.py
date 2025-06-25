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


def disk_galaxy(n, center, velocity, rng, m_core=1.0, m_disk=0.3, radius=1.0, eps=0.05):
    # Central heavy body plus a cold disk on (roughly) circular orbits.
    r = radius * np.sqrt(rng.uniform(0.02, 1.0, n))
    phi = rng.uniform(0, 2 * np.pi, n)
    pos = np.stack([r * np.cos(phi), r * np.sin(phi)], 1)
    m_enc = m_core + m_disk * (r / radius) ** 2
    v = np.sqrt(G * m_enc * r ** 2 / (r ** 2 + eps ** 2) ** 1.5)
    vel = np.stack([-v * np.sin(phi), v * np.cos(phi)], 1)
    pos = np.vstack([[0.0, 0.0], pos]) + center
    vel = np.vstack([[0.0, 0.0], vel]) + velocity
    mass = np.concatenate([[m_core], np.full(n, m_disk / n)])
    return pos, vel, mass


def clustered_blob(n, rng):
    pos = rng.standard_normal((n, 2)) / np.sqrt(1 + rng.exponential(1.0, (n, 1)))
    return pos, np.full(n, 1.0 / n)


def benchmark(quick, out="scaling.png"):
    rng = np.random.default_rng(1)
    eps = 0.01
    pos, mass = clustered_blob(1000 if quick else 2000, rng)
    ref = direct_accel(pos, mass, eps)
    thetas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    errs = []
    print("force accuracy, N =", len(pos))
    for th in thetas:
        a = bh_accel(pos, mass, th, eps)
        rel = np.linalg.norm(a - ref, axis=1) / np.linalg.norm(ref, axis=1)
        errs.append(np.median(rel))
        print(f"  theta={th:.1f}  median rel error {errs[-1]:.2e}  99th pct {np.percentile(rel, 99):.2e}")

    ns = [250, 500, 1000, 2000] if quick else [250, 500, 1000, 2000, 4000, 8000]
    t_direct, t_bh = [], []
    print("runtime (s), theta = 0.7")
    for n in ns:
        p, m = clustered_blob(n, rng)
        t0 = time.perf_counter(); direct_accel(p, m, eps); t1 = time.perf_counter()
        bh_accel(p, m, 0.7, eps); t2 = time.perf_counter()
        t_direct.append(t1 - t0); t_bh.append(t2 - t1)
        print(f"  N={n:5d}  direct {t_direct[-1]:.3f}  barnes-hut {t_bh[-1]:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.semilogy(thetas[1:], errs[1:], "o-")
    ax1.set(xlabel="theta", ylabel="median relative force error", title=f"Accuracy (N={len(pos)})")
    ax2.loglog(ns, t_direct, "o-", label="direct O(N²)")
    ax2.loglog(ns, t_bh, "s-", label="Barnes-Hut, θ=0.7")
    ax2.set(xlabel="N", ylabel="seconds per force evaluation", title="Runtime scaling")
    ax2.set_xticks(ns, [str(n) for n in ns])
    ax2.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax2.legend()
    for ax in (ax1, ax2):
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("wrote", out)


def render(pos, which, size=320, extent=5.0):
    img = np.zeros((size, size, 3))
    colours = [np.array([1.0, 0.55, 0.2]), np.array([0.35, 0.65, 1.0])]
    for g, col in enumerate(colours):
        p = pos[which == g]
        h, _, _ = np.histogram2d(p[:, 1], p[:, 0], bins=size, range=[[-extent, extent]] * 2)
        h = h + 0.4 * (np.roll(h, 1, 0) + np.roll(h, -1, 0) + np.roll(h, 1, 1) + np.roll(h, -1, 1))
        img += (1 - np.exp(-h / 0.6))[::-1, :, None] * col
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def collide(quick, theta, out="galaxies.gif"):
    rng = np.random.default_rng(0)
    n = 150 if quick else 500
    eps = 0.12
    # offset, prograde encounter: pericentre ~0.9, close enough to pull out tidal tails
    a = disk_galaxy(n, [-2.0, -1.25], [0.4, 0.0], rng, eps=eps)
    b = disk_galaxy(n, [2.0, 1.25], [-0.4, 0.0], rng, eps=eps)
    pos, vel, mass = (np.concatenate(x) for x in zip(a, b))
    which = np.repeat([0, 1], n + 1)
    vel -= (mass[:, None] * vel).sum(0) / mass.sum()  # centre-of-mass frame

    accel = lambda p: bh_accel(p, mass, theta, eps)
    dt, steps, every = 0.02, 600, 4
    acc = accel(pos)
    e0 = energy(pos, vel, mass, eps)
    frames = []
    t0 = time.perf_counter()
    for s in range(steps):
        if s % every == 0:
            frames.append(render(pos, which))
        acc = leapfrog_step(pos, vel, acc, dt, accel)
    e1 = energy(pos, vel, mass, eps)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=50, loop=0)
    print(f"collision: {len(pos)} bodies, {steps} steps in {time.perf_counter() - t0:.1f}s, "
          f"relative energy change {abs(e1 - e0) / abs(e0):.2e}, wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=0.7)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    benchmark(args.quick)
    collide(args.quick, args.theta)


if __name__ == "__main__":
    main()
