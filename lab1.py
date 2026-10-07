import numpy as np


def canonical_form(c, A, b, signs):
    m, n = A.shape
    basis = []
    art_idx = []
    extra = 0

    for i in range(m):
        if signs[i] == '<=':
            extra += 1
        elif signs[i] == '>=':
            extra += 2
        elif signs[i] == '=':
            extra += 1

    total_vars = n + extra
    A_full = np.zeros((m, total_vars))
    b_full = np.array(b, dtype=float)
    c_full = np.zeros(total_vars)
    c_full[:n] = c

    col = n
    for i in range(m):
        A_full[i, :n] = A[i]
        if signs[i] == '<=':
            A_full[i, col] = 1
            col += 1
        elif signs[i] == '>=':
            A_full[i, col] = -1
            col += 1
            A_full[i, col] = 1
            basis.append(col)
            art_idx.append(col)
            col += 1
        elif signs[i] == '=':
            A_full[i, col] = 1
            basis.append(col)
            art_idx.append(col)
            col += 1

    col = n
    for i in range(m):
        if signs[i] == '<=':
            basis.append(col)
            col += 1
        elif signs[i] == '>=':
            col += 2
        elif signs[i] == '=':
            col += 1

    return c_full, A_full, b_full, basis, art_idx, n


def rebuild_W(A, b, basis, art_idx, n):
    m = A.shape[0]
    W = np.zeros(n + 1)
    for i in range(m):
        if basis[i] in art_idx:
            W[:n] -= A[i]
            W[-1] += b[i]
    for j in basis:
        W[j] = 0.0
    return W


def rebuild_Z(A, b, basis, c, n):
    m = A.shape[0]
    Z = np.zeros(n + 1)

    Z[:n] = c[:n]

    for i in range(m):
        j = basis[i]
        if j < n:
            factor = c[j]
            Z[:n] -= factor * A[i]
            Z[-1] += factor * b[i]

    for j in basis:
        if j < n:
            Z[j] = 0.0

    return Z


def simplex_step(A, b, c, basis, maximize=True):
    m, n = A.shape

    if maximize:
        candidates = [j for j in range(n) if c[j] > 1e-9]
        if not candidates:
            return A, b, c, basis, True, False
        pivot_col = max(candidates, key=lambda j: c[j])
    else:
        candidates = [j for j in range(n) if c[j] < -1e-9]
        if not candidates:
            return A, b, c, basis, True, False
        pivot_col = min(candidates, key=lambda j: c[j])

    ratios = []
    for i in range(m):
        if A[i, pivot_col] > 1e-9:
            ratios.append((b[i] / A[i, pivot_col], i))
    if not ratios:
        return A, b, c, basis, False, True

    pivot_row = min(ratios, key=lambda x: x[0])[1]
    pivot_elem = A[pivot_row, pivot_col]

    A[pivot_row] /= pivot_elem
    b[pivot_row] /= pivot_elem

    for i in range(m):
        if i != pivot_row:
            factor = A[i, pivot_col]
            A[i] -= factor * A[pivot_row]
            b[i] -= factor * b[pivot_row]

    factor = c[pivot_col]
    c -= factor * np.append(A[pivot_row], b[pivot_row])

    basis[pivot_row] = pivot_col

    return A, b, c, basis, False, False


def solve_lp(c, A, b, signs):
    c_full, A_full, b_full, basis, art_idx, n_orig = canonical_form(c, A, b, signs)
    m, n = A_full.shape

    if art_idx:
        A_aux = A_full.copy()
        b_aux = b_full.copy()
        basis_aux = basis.copy()
        W_row = rebuild_W(A_aux, b_aux, basis_aux, art_idx, n)

        for _ in range(200):
            A_aux, b_aux, W_row, basis_aux, optimal, unbounded = simplex_step(
                A_aux, b_aux, W_row, basis_aux, maximize=False
            )
            if unbounded:
                return None, None
            if optimal:
                break
            W_row = rebuild_W(A_aux, b_aux, basis_aux, art_idx, n)

        W_min = W_row[-1]
        if abs(W_min) > 1e-6:
            return None, None

        for i in range(m):
            if basis_aux[i] in art_idx:
                for j in range(n):
                    if j not in art_idx and abs(A_aux[i, j]) > 1e-6:
                        pivot_elem = A_aux[i, j]
                        A_aux[i] /= pivot_elem
                        b_aux[i] /= pivot_elem
                        for k in range(m):
                            if k != i:
                                factor = A_aux[k, j]
                                A_aux[k] -= factor * A_aux[i]
                                b_aux[k] -= factor * b_aux[i]
                        basis_aux[i] = j
                        break

        keep = [j for j in range(n) if j not in art_idx]
        A_aux = A_aux[:, keep]
        c_keep = c_full[keep]
        n = len(keep)
        basis = [keep.index(j) for j in basis_aux]
    else:
        A_aux = A_full.copy()
        b_aux = b_full.copy()
        basis = basis.copy()
        n = A_full.shape[1]
        c_keep = c_full

    Z_row = rebuild_Z(A_aux, b_aux, basis, c_keep, n)

    for _ in range(200):
        A_aux, b_aux, Z_row, basis, optimal, unbounded = simplex_step(
            A_aux, b_aux, Z_row, basis, maximize=True
        )
        if unbounded:
            return None, None
        if optimal:
            break
        Z_row = rebuild_Z(A_aux, b_aux, basis, c_keep, n)

    x_opt = np.zeros(n_orig)
    for i, j in enumerate(basis):
        if j < n_orig:
            x_opt[j] = b_aux[i]

    Z_opt = 0.0
    for j in range(n_orig):
        Z_opt += c[j] * x_opt[j]

    return x_opt, Z_opt


if __name__ == "__main__":
    c = np.array([4, 1, 2, 3], dtype=float)
    A = np.array([
        [1, 1, 1, 0],
        [0, 1, 2, 1],
        [1, 0, 0, 1]
    ], dtype=float)
    b = np.array([9, 7, 3], dtype=float)
    signs = ['<=', '=', '>=']

    x_opt, Z_opt = solve_lp(c, A, b, signs)

    if x_opt is not None:
        print(f"x = {x_opt}")
        print(f"Z = {Z_opt:.6f}")
    else:
        print("Решение не найдено")