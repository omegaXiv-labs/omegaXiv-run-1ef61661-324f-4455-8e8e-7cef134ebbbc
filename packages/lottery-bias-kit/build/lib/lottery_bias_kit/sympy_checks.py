from __future__ import annotations

import sympy as sp


def compute_sympy_invariants() -> dict[str, str]:
    """Return symbolic identities used by the partial-identification methodology."""
    delta, gamma, gamma_budget = sp.symbols("Delta gamma Gamma", real=True)
    abs_term = sp.Abs(gamma) * gamma_budget

    z = sp.symbols("z", real=True)
    sigma = 1 / (1 + sp.exp(-z))
    logistic_second = sp.simplify(
        sp.diff(sp.log(1 + sp.exp(z)), z, 2) - sigma * (1 - sigma)
    )

    lhs = sp.Or(delta - abs_term > 0, delta + abs_term < 0)
    rhs = sp.Abs(delta) > abs_term
    sign_equivalence = sp.simplify_logic(sp.Equivalent(lhs, rhs), form="cnf")

    by_i, m, q = sp.symbols("i m q", positive=True)
    h_m = sp.symbols("H_m", positive=True)
    t_i = by_i * q / (m * h_m)
    monotone_increment = sp.simplify((by_i + 1) * q / (m * h_m) - t_i)

    return {
        "logistic_second_derivative_identity": str(logistic_second),
        "sign_identification_equivalence_cnf": str(sign_equivalence),
        "by_monotone_increment": str(monotone_increment),
    }
