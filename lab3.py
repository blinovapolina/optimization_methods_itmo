import itertools
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import sympy as sp


@dataclass
class LagrangeResult:
    irregular_points: list[dict] = field(default_factory=list)

    f: sp.Expr | None = None
    constraints: list[sp.Expr] = field(default_factory=list)
    variables: tuple[sp.Symbol, ...] = ()

    multipliers: tuple[sp.Symbol, ...] = ()
    lagrangian: sp.Expr | None = None

    partials: dict[str, sp.Expr] = field(default_factory=dict)

    system: list[sp.Expr] = field(default_factory=list)

    candidates: list[dict] = field(default_factory=list)

    valid_candidates: list[dict] = field(default_factory=list)
    rejected_candidates: list[dict] = field(default_factory=list)

    classified: list[dict] = field(default_factory=list)

    values: list[tuple[float, dict]] = field(default_factory=list)
    f_min: float | None = None
    f_max: float | None = None
    argmin: list[dict] = field(default_factory=list)
    argmax: list[dict] = field(default_factory=list)

    answer: str = ""


def _as_expr_list(exprs: Sequence[sp.Expr] | sp.Expr) -> list[sp.Expr]:
    if isinstance(exprs, (list, tuple)):
        return [sp.sympify(e) for e in exprs]
    return [sp.sympify(exprs)]


def _grad(expr: sp.Expr, variables: Sequence[sp.Symbol]) -> list[sp.Expr]:
    return [sp.diff(expr, v) for v in variables]


def _on_constraints(point: dict, constraints: list[sp.Expr],
                    tol: float = 1e-9) -> bool:
    for g in constraints:
        val = complex(g.subs(point).evalf())
        if abs(val) > tol:
            return False
    return True


def _numeric(expr: sp.Expr, point: dict) -> float:
    return float(sp.re(expr.subs(point).evalf()))


def _fmt_point(point: dict, keys: Sequence[sp.Symbol] | None = None) -> str:
    items = point.items() if keys is None else [(k, point[k]) for k in keys]
    return "{" + ", ".join(f"{k}: {v}" for k, v in items) + "}"


def step1_irregular_points(
    constraints: list[sp.Expr],
    variables: Sequence[sp.Symbol],
) -> list[dict]:
    m = len(constraints)
    if m == 0:
        return []

    irregular: list[dict] = []

    if m == 1:
        g = constraints[0]
        zero_system = [sp.Eq(c, 0) for c in _grad(g, variables)]
        for sol in sp.solve(zero_system, variables, dict=True):
            full = {v: sp.simplify(sol.get(v, v)) for v in variables}
            if all(val.is_number for val in full.values()) \
                    and _on_constraints(full, constraints):
                irregular.append(full)
        return irregular

    jac = sp.Matrix([_grad(g, variables) for g in constraints])
    minors = []
    for rows in itertools.combinations(range(m), m):
        for cols in itertools.combinations(range(len(variables)), m):
            minors.append(sp.simplify(jac.extract(rows, cols).det()))
    equations = [sp.Eq(mi, 0) for mi in minors if mi != 0]
    if not equations:
        return []
    equations += [sp.Eq(g, 0) for g in constraints]
    for sol in sp.solve(equations, variables, dict=True):
        full = {v: sp.simplify(sol.get(v, v)) for v in variables}
        if not all(val.is_number for val in full.values()):
            continue
        if not _on_constraints(full, constraints):
            continue
        if jac.subs(full).rank() < m:
            irregular.append(full)
    return irregular


def step2_formulate(
    f: sp.Expr,
    constraints: Sequence[sp.Expr] | sp.Expr,
    variables: Sequence[sp.Symbol] | None,
) -> tuple[sp.Expr, list[sp.Expr], tuple[sp.Symbol, ...]]:
    f = sp.sympify(f)
    constraints = _as_expr_list(constraints)
    if variables is None:
        symbols = set(f.free_symbols)
        for g in constraints:
            symbols |= g.free_symbols
        variables = tuple(sorted(symbols, key=lambda s: s.name))
    return f, constraints, tuple(variables)


def step3_lagrangian(
    f: sp.Expr,
    constraints: list[sp.Expr],
    multipliers: Sequence[sp.Symbol] | None = None,
) -> tuple[sp.Expr, tuple[sp.Symbol, ...]]:
    if multipliers is None:
        multipliers = sp.symbols(f"lambda1:{len(constraints)+1}")
    multipliers = tuple(multipliers)
    L = sp.expand(f + sum(lam * g for lam, g in zip(multipliers, constraints)))
    return L, multipliers


def step4_partials(
    L: sp.Expr,
    variables: Sequence[sp.Symbol],
    multipliers: Sequence[sp.Symbol],
) -> dict[str, sp.Expr]:
    partials: dict[str, sp.Expr] = {}
    for v in variables:
        partials[f"dL/d{v}"] = sp.simplify(sp.diff(L, v))
    for lam in multipliers:
        partials[f"dL/d{lam}"] = sp.simplify(sp.diff(L, lam))
    return partials


def step5_system(partials: dict[str, sp.Expr]) -> list[sp.Expr]:
    return [expr for expr in partials.values()]


def step6_solve(
    system: list[sp.Expr],
    variables: Sequence[sp.Symbol],
    multipliers: Sequence[sp.Symbol],
) -> list[dict]:
    unknowns = list(variables) + list(multipliers)
    equations = [sp.Eq(e, 0) for e in system]
    raw = sp.solve(equations, unknowns, dict=True)
    points = []
    for sol in raw:
        full = {u: sp.simplify(sol.get(u, u)) for u in unknowns}
        if all(val.is_number for val in full.values()):
            points.append(full)

    unique, seen = [], set()
    for p in points:
        key = tuple((str(u), sp.nsimplify(p[u])) for u in unknowns)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def step7_check(
    candidates: list[dict],
    constraints: list[sp.Expr],
) -> tuple[list[dict], list[dict]]:
    valid, rejected = [], []
    for p in candidates:
        (valid if _on_constraints(p, constraints) else rejected).append(p)
    return valid, rejected



def _bordered_hessian(
    L: sp.Expr,
    constraints: list[sp.Expr],
    variables: Sequence[sp.Symbol],
    point: dict,
) -> tuple[np.ndarray, str]:
    n, m = len(variables), len(constraints)

    J = np.array([
        [float(sp.re(sp.diff(g, v).subs(point).evalf())) for v in variables]
        for g in constraints
    ], dtype=float)
    H_xx = np.array([
        [float(sp.re(sp.diff(L, vi, vj).subs(point).evalf())) for vj in variables]
        for vi in variables
    ], dtype=float)

    H = np.zeros((m + n, m + n), dtype=float)
    H[:m, m:] = J
    H[m:, :m] = J.T
    H[m:, m:] = H_xx

    minors = []
    for k in range(2 * m + 1, m + n + 1):
        minors.append(float(np.linalg.det(H[:k, :k])))

    if not minors or any(abs(d) < 1e-10 for d in minors):
        return H, "unknown"

    if m == 1:
        return H, ("max" if minors[0] > 0 else "min")

    signs = [np.sign(d) for d in minors]
    if all(((-1) ** m) * s > 0 for s in signs):
        return H, "min"
    if all(((-1) ** k) * s > 0
           for k, s in zip(range(2 * m + 1, m + n + 1), signs)):
        return H, "max"
    return H, "saddle"


def step8_classify(
    valid_candidates: list[dict],
    f: sp.Expr,
    L: sp.Expr,
    constraints: list[sp.Expr],
    variables: Sequence[sp.Symbol],
) -> list[dict]:
    classified = []
    for p in valid_candidates:
        point_vars = {v: p[v] for v in variables}
        f_val = _numeric(f, point_vars)
        H, verdict = _bordered_hessian(L, constraints, variables, p)
        classified.append({
            "point": p,
            "f_value": f_val,
            "hessian": H,
            "hessian_verdict": verdict,
        })
    return classified


def step9_optimal(
    classified: list[dict],
    variables: Sequence[sp.Symbol],
) -> dict:
    values = [(c["f_value"], c["point"]) for c in classified]
    if not values:
        return {"values": [], "f_min": None, "f_max": None,
                "argmin": [], "argmax": []}
    f_min = min(v for v, _ in values)
    f_max = max(v for v, _ in values)
    tol = 1e-9
    argmin = [p for v, p in values if abs(v - f_min) < tol]
    argmax = [p for v, p in values if abs(v - f_max) < tol]
    return {"values": values, "f_min": f_min, "f_max": f_max,
            "argmin": argmin, "argmax": argmax}


def step10_answer(
    optimal: dict,
    variables: Sequence[sp.Symbol],
) -> str:
    def pts(lst):
        return "; ".join(_fmt_point(p, variables) for p in lst)

    if optimal["f_min"] is None:
        return "Допустимых стационарных точек нет; условный экстремум не найден."

    lines = []
    lines.append(f"f_min = {optimal['f_min']} в точках: {pts(optimal['argmin'])}")
    lines.append(f"f_max = {optimal['f_max']} в точках: {pts(optimal['argmax'])}")
    return "\n".join(lines)


def solve_lagrange(
    f: sp.Expr,
    constraints: Sequence[sp.Expr] | sp.Expr,
    variables: Sequence[sp.Symbol] | None = None,
    *,
    verbose: bool = True,
) -> LagrangeResult:
    res = LagrangeResult()

    def log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    log("=" * 72)

    f_expr, constraints_expr, vars_tuple = step2_formulate(f, constraints, variables)
    res.f, res.constraints, res.variables = f_expr, constraints_expr, vars_tuple

    res.irregular_points = step1_irregular_points(constraints_expr, vars_tuple)
    log("Шаг 1. Нерегулярные точки:",
        [_fmt_point(p, vars_tuple) for p in res.irregular_points] or "нет")

    log("Шаг 2. Постановка задачи")
    log(f"   f({', '.join(map(str, vars_tuple))}) = {f_expr}")
    for i, g in enumerate(constraints_expr, 1):
        log(f"   g{i} = {g} = 0")

    res.lagrangian, res.multipliers = step3_lagrangian(f_expr, constraints_expr)
    log(f"Шаг 3. Функция Лагранжа: L = {res.lagrangian}")

    res.partials = step4_partials(res.lagrangian, vars_tuple, res.multipliers)
    log("Шаг 4. Частные производные:")
    for name, expr in res.partials.items():
        log(f"   {name} = {expr}")

    res.system = step5_system(res.partials)
    log("Шаг 5. Система необходимых условий:")
    for expr in res.system:
        log(f"   {expr} = 0")

    res.candidates = step6_solve(res.system, vars_tuple, res.multipliers)
    log("Шаг 6. Кандидаты (стационарные точки):")
    for p in res.candidates:
        log(f"   {_fmt_point(p)}")
    if not res.candidates:
        log("   нет решений системы")

    res.valid_candidates, res.rejected_candidates = step7_check(
        res.candidates, constraints_expr
    )
    log(f"Шаг 7. Допустимых кандидатов: {len(res.valid_candidates)}"
        f" (отброшено: {len(res.rejected_candidates)})")
    for p in res.rejected_candidates:
        log(f"   отброшена {_fmt_point(p)} — не удовлетворяет ограничениям")

    res.classified = step8_classify(
        res.valid_candidates, f_expr, res.lagrangian, constraints_expr, vars_tuple
    )
    log("Шаг 8. Характер точек:")
    for c in res.classified:
        log(f"   {_fmt_point(c['point'], vars_tuple)}: "
            f"f = {c['f_value']}, "
            f"характер = {c['hessian_verdict']}")

    optimal = step9_optimal(res.classified, vars_tuple)
    res.values = optimal["values"]
    res.f_min, res.f_max = optimal["f_min"], optimal["f_max"]
    res.argmin, res.argmax = optimal["argmin"], optimal["argmax"]
    log("Шаг 9. Значения f:")
    for v, p in res.values:
        log(f"   f{_fmt_point(p, vars_tuple)} = {v}")
    log(f"   глобальный min = {res.f_min}, max = {res.f_max}")

    res.answer = step10_answer(optimal, vars_tuple)
    log("Шаг 10. Ответ:")
    for line in res.answer.splitlines():
        log("   " + line)
    log("=" * 72)

    return res

if __name__ == "__main__":
    x, y = sp.symbols("x y")
    f = x**2 + 2 * y**2
    g = x**2 + y**2 - 9

    res = solve_lagrange(f, g, (x, y), verbose=True)