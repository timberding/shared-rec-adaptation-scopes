"""CPU checks of a proposed KL-to-membership certificate; no model training."""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import rel_entr


def radius(p, k, m):
    """Return the exact unrestricted-distribution boundary cost and minimizer.

    shortcut: float64 arithmetic is a numerical probe, not a rounded certificate;
    use interval arithmetic if these values become a deployment acceptance gate.
    """
    p = np.asarray(p, dtype=float)
    if (p.ndim != 1 or np.any(~np.isfinite(p)) or np.any(p <= 0)
            or not np.isclose(p.sum(), 1, atol=1e-12, rtol=0)
            or np.any(np.diff(p) > 0) or not 1 <= k < len(p)
            or not 1 <= m <= min(k, len(p) - k) or p[k-1] <= p[k]):
        raise ValueError('positive sorted distribution, strict K boundary, valid K,m required')
    inside = p[k-m:k]
    outside = p[k:k+m]
    lo, hi = float(outside[-1]), float(inside[0])
    for _ in range(100):
        t = (lo + hi) / 2
        removed = np.maximum(inside-t, 0).sum()
        added = np.maximum(t-outside, 0).sum()
        if removed > added:
            lo = t
        else:
            hi = t
    t = (lo + hi) / 2
    q = p.copy()
    q[k-m:k] = np.minimum(inside, t)
    q[k:k+m] = np.maximum(outside, t)
    assert abs(q.sum()-1) < 1e-12
    return float(rel_entr(p, q).sum()), q, t


def exhaustive_minimum(p, k, m):
    """Independent convex solves for every eligible Top-K set, no clipping code."""
    n = len(p)
    best = np.inf
    solves = 0
    for selected in itertools.combinations(range(n), k):
        if sum(i >= k for i in selected) < m:
            continue
        other = sorted(set(range(n)) - set(selected))
        a = np.zeros((k*(n-k), n))
        for row, (i, j) in enumerate(itertools.product(selected, other)):
            a[row, i], a[row, j] = 1, -1
        result = minimize(
            lambda q: float(rel_entr(p, q).sum()), np.full(n, 1/n),
            jac=lambda q: -p/q, method='SLSQP',
            bounds=[(1e-10, 1)]*n,
            constraints=[{'type': 'eq', 'fun': lambda q: q.sum()-1,
                          'jac': lambda q: np.ones(n)},
                         {'type': 'ineq', 'fun': lambda q: a@q,
                          'jac': lambda q: a}],
            options={'ftol': 1e-12, 'maxiter': 800})
        assert result.success, result.message
        assert np.min(a@result.x) > -1e-8 and abs(result.x.sum()-1) < 1e-8
        best = min(best, result.fun)
        solves += 1
    return best, solves


def run_checks():
    rng = np.random.default_rng(20261116)
    exhaustive, solves, errors = 0, 0, []
    for n in range(3, 8):
        for _ in range(3):
            p = np.sort(rng.uniform(.05, 1, n))[::-1]
            p /= p.sum()
            for k in range(1, n):
                last = 0
                for m in range(1, min(k, n-k)+1):
                    r, q, _ = radius(p, k, m)
                    e, count = exhaustive_minimum(p, k, m)
                    errors.append(abs(e-r)); solves += count; exhaustive += 1
                    assert abs(e-r) < 2e-8, (p, k, m, e, r)
                    assert r >= last-1e-12
                    last = r
                    # Move infinitesimally inside the desired strict-order cone.
                    for eps in [1e-7, 1e-9]:
                        qp = q.copy()
                        qp[k-m:k] -= eps
                        qp[k:k+m] += eps
                        assert qp.min() > 0
                        selected = np.argsort(-qp, kind='stable')[:k]
                        assert sum(i >= k for i in selected) >= m
                        assert float(rel_entr(p, qp).sum()) >= r-1e-12
    # Arbitrary changes cannot cross a replacement boundary below its cost.
    probes = 0
    for _ in range(1200):
        n = int(rng.integers(4, 41)); k = int(rng.integers(1, n))
        p = np.sort(rng.dirichlet(np.ones(n)))[::-1]
        q = rng.dirichlet(np.ones(n))
        replaced = sum(i >= k for i in np.argsort(-q)[:k])
        divergence = float(rel_entr(p, q).sum())
        if replaced:
            r, _, _ = radius(p, k, replaced)
            assert divergence >= r-1e-12
        probes += 1
    # Equality is not a safe membership guarantee; tie policy can select an entrant.
    # The canonical crossing condition permits MORE than m entrants; do not force
    # the first K-m insiders into a Top-K set for an arbitrary feasible q.
    arbitrary_q = np.array([.2, .1, .4, .3])
    assert arbitrary_q[2] >= arbitrary_q[1]
    assert set(np.argsort(-arbitrary_q)[:2]) == {2, 3}
    assert arbitrary_q[0] < arbitrary_q[3]
    p = np.array([.6, .4]); r, q, _ = radius(p, 1, 1)
    assert np.allclose(q, [.5, .5])
    assert np.isclose(r, .6*np.log(1.2)+.4*np.log(.8))
    # Pooling all 2m entries overstates the KL cost when not all constraints bind.
    p = np.array([.4, .15, .14, .13, .10, .08])
    r, q, t = radius(p, 2, 2)
    uniform_block = p.copy(); uniform_block[:4] = p[:4].mean()
    naive = float(rel_entr(p, uniform_block).sum())
    assert q[1] == p[1] and r < naive-1e-3
    # Same mean KL is compatible with zero churn or complete replacement.
    eps = 1e-3; k = 10
    pflat = np.r_[np.full(k, (1+eps)/(2*k)), np.full(k, (1-eps)/(2*k))]
    qflat = np.r_[pflat[k:], pflat[:k]]
    small_kl = float(rel_entr(pflat, qflat).sum())
    assert sum(i >= k for i in np.argsort(-qflat)[:k]) == k
    assert np.isclose(small_kl, eps*np.log((1+eps)/(1-eps)))
    # Renormalized teacher-Top-L KL can be exactly zero while an omitted item enters.
    pt = np.array([.45, .35, .15, .05]); qt = np.array([.045, .035, .015, .905])
    assert abs(rel_entr(pt[:3]/pt[:3].sum(), qt[:3]/qt[:3].sum()).sum()) < 1e-15
    assert set(np.argsort(-pt)[:2]) != set(np.argsort(-qt)[:2])
    # Cohort cost bound: every subset obeys the cheapest-radii budget bound.
    costs = np.array([.01, .03, .04, .08, .2])
    for mask in itertools.product([0, 1], repeat=len(costs)):
        n_changed = sum(mask); budget = float(costs@mask)
        allowed = max(j for j in range(len(costs)+1)
                      if costs[:j].sum() <= budget+1e-12)
        assert n_changed <= allowed
    result = {
        'status': 'PASS', 'purpose': 'mathematical CPU probes, no recommender evidence',
        'exhaustive_cases': exhaustive, 'independent_convex_solves': solves,
        'maximum_radius_error': max(errors), 'random_distribution_probes': probes,
        'counterexamples': {
            'all_2m_pooling': {'p': p.tolist(), 'k': 2, 'm': 2, 'threshold': t,
                             'q': q.tolist(), 'exact_radius': r, 'naive_radius': naive},
            'tiny_KL_complete_replacement': {'K': k, 'KL': small_kl, 'replaced': k},
            'truncated_KL_zero': {'p': pt.tolist(), 'q': qt.tolist(), 'K': 2, 'L': 3}},
        'limitations': 'SLSQP/random tests do not prove theorems; no real logits read.'}
    Path(__file__).with_name('NUMERIC_CHECKS.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run_checks()
