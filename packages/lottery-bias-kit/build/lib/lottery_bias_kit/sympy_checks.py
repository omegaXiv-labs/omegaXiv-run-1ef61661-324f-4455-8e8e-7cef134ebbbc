from __future__ import annotations

import sympy as sp


def compute_sympy_invariants() -> dict[str, str]:
    """Return symbolic identities used by staged H4/H5 methodology."""
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

    delta_ni, delta_sup = sp.symbols("delta_NI delta_SUP", nonnegative=True)
    staged_gate_implication = sp.simplify_logic(
        sp.Implies(sp.Symbol("Delta") <= -delta_sup, sp.Symbol("Delta") <= delta_ni),
        form="cnf",
    )

    w1, w2, d11, d12, d21, d22, alpha1, alpha2 = sp.symbols(
        "w1 w2 d11 d12 d21 d22 alpha1 alpha2",
        nonnegative=True,
    )
    lhs = alpha1 * (w1 * d11 + w2 * d12) + alpha2 * (w1 * d21 + w2 * d22)
    rhs = w1 * (alpha1 * d11 + alpha2 * d21) + w2 * (alpha1 * d12 + alpha2 * d22)
    conic_identity = sp.simplify(lhs - rhs)

    return {
        "logistic_second_derivative_identity": str(logistic_second),
        "sign_identification_equivalence_cnf": str(sign_equivalence),
        "by_monotone_increment": str(monotone_increment),
        "staged_gate_implication_cnf": str(staged_gate_implication),
        "conic_combination_inner_product_identity": str(conic_identity),
    }
