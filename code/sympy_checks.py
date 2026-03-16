from __future__ import annotations

from pathlib import Path

import sympy as sp


def run_sympy_validation(report_path: Path) -> dict:
    Delta, gamma, Gamma = sp.symbols("Delta gamma Gamma", real=True)
    a = sp.Abs(gamma) * Gamma

    z = sp.symbols("z", real=True)
    sigma = 1 / (1 + sp.exp(-z))
    logistic_second = sp.simplify(sp.diff(sp.log(1 + sp.exp(z)), z, 2) - sigma * (1 - sigma))

    cond_left = sp.Or(Delta - a > 0, Delta + a < 0)
    cond_right = sp.Abs(Delta) > a
    equivalence = sp.simplify_logic(sp.Equivalent(cond_left, cond_right), form="cnf")

    by_i, m, q = sp.symbols("i m q", positive=True)
    Hm = sp.symbols("H_m", positive=True)
    t_i = by_i * q / (m * Hm)
    monotone_step = sp.simplify((by_i + 1) * q / (m * Hm) - t_i)

    delta_ni, delta_sup = sp.symbols("delta_NI delta_SUP", nonnegative=True)
    gate_implication = sp.simplify_logic(
        sp.Implies(sp.Symbol("Delta") <= -delta_sup, sp.Symbol("Delta") <= delta_ni), form="cnf"
    )

    w1, w2, d11, d12, d21, d22, alpha1, alpha2 = sp.symbols(
        "w1 w2 d11 d12 d21 d22 alpha1 alpha2", nonnegative=True
    )
    lhs = alpha1 * (w1 * d11 + w2 * d12) + alpha2 * (w1 * d21 + w2 * d22)
    rhs = w1 * (alpha1 * d11 + alpha2 * d21) + w2 * (alpha1 * d12 + alpha2 * d22)
    conic_identity = sp.simplify(lhs - rhs)

    payload = {
        "logistic_second_derivative_identity": str(logistic_second),
        "sign_identification_equivalence_cnf": str(equivalence),
        "by_monotone_increment": str(monotone_step),
        "staged_gate_implication_cnf": str(gate_implication),
        "conic_combination_inner_product_identity": str(conic_identity),
        "notes": [
            "logistic_second_derivative_identity should simplify to 0.",
            "sign-identification equivalence encodes C3 iff condition.",
            "BY increment should be positive for q,m,H_m > 0.",
            "staged_gate_implication encodes C4 Stage-B => Stage-A safety.",
            "conic combination identity supports C5 impossibility argument algebra.",
        ],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(sp.pretty(payload), encoding="utf-8")
    return payload
