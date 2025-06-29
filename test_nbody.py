import numpy as np

import nbody


def random_bodies(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 2)), rng.uniform(0.5, 1.5, n)


def test_theta_zero_matches_direct_sum():
    pos, mass = random_bodies(300)
    for eps in (0.0, 0.05):
        np.testing.assert_allclose(nbody.bh_accel(pos, mass, theta=0.0, eps=eps),
                                   nbody.direct_accel(pos, mass, eps), rtol=1e-9, atol=1e-12)


def test_theta_controls_accuracy():
    pos, mass = random_bodies(500, seed=1)
    ref = nbody.direct_accel(pos, mass, 0.01)
    err = [np.median(np.linalg.norm(nbody.bh_accel(pos, mass, th, 0.01) - ref, axis=1)
                     / np.linalg.norm(ref, axis=1)) for th in (0.3, 0.6, 1.0)]
    assert err[0] < err[1] < err[2] < 0.1


def test_momentum_conserved():
    # direct-sum forces (theta=0) are pairwise antisymmetric, so momentum is exact
    pos, mass = random_bodies(200, seed=2)
    vel = np.random.default_rng(3).standard_normal((200, 2)) * 0.3
    p0 = (mass[:, None] * vel).sum(0)
    accel = lambda q: nbody.bh_accel(q, mass, theta=0.0, eps=0.05)
    a = accel(pos)
    for _ in range(50):
        a = nbody.leapfrog_step(pos, vel, a, 0.01, accel)
    scale = (mass[:, None] * np.abs(vel)).sum()
    assert np.abs((mass[:, None] * vel).sum(0) - p0).max() < 1e-12 * scale


def test_tree_net_force_nearly_zero():
    # monopole approximations break Newton's third law, but only slightly
    pos, mass = random_bodies(200, seed=2)
    a = nbody.bh_accel(pos, mass, theta=0.7, eps=0.05)
    net = np.linalg.norm((mass[:, None] * a).sum(0))
    assert net < 1e-2 * (mass * np.linalg.norm(a, axis=1)).sum()


def test_leapfrog_two_body_energy_drift_small():
    # eccentric (e=0.5) Kepler orbit, 20 periods at ~200 steps per period
    mass = np.array([1.0, 1e-3])
    pos = np.array([[0.0, 0.0], [1.0, 0.0]])
    vel = np.array([[0.0, 0.0], [0.0, np.sqrt(1.5 * mass.sum())]])  # r=1 is pericentre: v^2 = GM(1+e)/r
    vel -= (mass[:, None] * vel).sum(0) / mass.sum()
    accel = lambda q: nbody.direct_accel(q, mass)
    e0 = nbody.energy(pos, vel, mass)
    a = accel(pos)
    period = 2 * np.pi * (1 / (1 - 0.5)) ** 1.5 / np.sqrt(mass.sum())  # semi-major axis a = r_p/(1-e)
    dt = period / 200
    energies = []
    for _ in range(20 * 200):
        a = nbody.leapfrog_step(pos, vel, a, dt, accel)
        energies.append(nbody.energy(pos, vel, mass))
    err = np.abs(np.array(energies) - e0) / abs(e0)
    assert err.max() < 1e-2
    # symplectic: the error oscillates instead of growing, so the last orbit is no worse than the first
    assert err[-200:].max() < 1.1 * err[:200].max()
