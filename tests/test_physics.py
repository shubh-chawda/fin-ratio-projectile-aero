import math

from src.fit_drag_model import simulate_range

G = 9.81


def test_no_drag_matches_ballistic_range():
    # For theta=45°, ideal range is R = v^2 / g
    v0 = 10.0
    theta = 45.0
    expected = (v0 * v0) / G

    got = simulate_range(v0, theta, k_eff=0.0, dt=1e-3, t_max=10.0)

    # allow small numerical error (RK4 + ground-cross interpolation)
    assert abs(got - expected) / expected < 0.01  # < 1%


def test_drag_reduces_range():
    v0 = 10.0
    theta = 45.0

    r0 = simulate_range(v0, theta, k_eff=0.0, dt=1e-3, t_max=10.0)
    r1 = simulate_range(v0, theta, k_eff=0.05, dt=1e-3, t_max=10.0)

    assert r1 < r0


def test_range_monotone_in_drag_strength():
    v0 = 10.0
    theta = 45.0

    r_small = simulate_range(v0, theta, k_eff=0.02, dt=1e-3, t_max=10.0)
    r_big = simulate_range(v0, theta, k_eff=0.08, dt=1e-3, t_max=10.0)

    assert r_big < r_small
