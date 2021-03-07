import numpy as np
from scipy import linalg as la

from .matrix import Matrix, MatrixDecomposition
from . import EPS


def rd_lu(A: Matrix):
    """
    Returns MatrixDecomposition with:
    X = product of permutation and lower triagonal matrix
    Y = product of permutation and upper triagonal matrix
    R = diagonal matrix
    """

    # Helper Functions
    def transpose(n, a, b):
        p = np.arange(n)
        p[a] = b
        p[b] = a
        return Matrix.perm(p)

    def elim_row(n, r, b_orig):
        b = b_orig.copy()
        """returns nxn Gauss elimination matrix"""
        E = Matrix.id(n)
        b[:r, :] = 0
        E.set_col(r, b)
        E[r, r] = 1
        return E

    def elim_col(n, r, b_orig):
        """returns nxn Gauss elimination matrix"""
        b = b_orig.copy()
        E = Matrix.id(n)
        b[:, :r] = 0
        E.set_row(r, b)
        E[r, r] = 1
        return E

    A = Matrix(A)
    m, n = A.shape

    X = Matrix.id(m)
    Xi = Matrix.id(m)
    R = A
    Y = Matrix.id(n)
    Yi = Matrix.id(n)
    for r in range(min(m, n)):
        # Pivot selection
        M = np.abs(R[r:, r:])
        pivot = np.unravel_index(np.argmax(M, axis=None), M.shape)
        pivot = [pivot[0] + r, pivot[1] + r]
        d = R[pivot[0], pivot[1]]
        if abs(d) < EPS:
            break
        P = transpose(m, r, pivot[0])
        Q = transpose(n, r, pivot[1])
        R = P @ R @ Q
        X = X @ P.T
        Xi = P @ Xi
        Y = Q.T @ Y
        Yi = Yi @ Q

        # Column elimination
        b = -R.get_col(r) / d
        E = elim_row(m, r, b)
        Ei = elim_row(m, r, -b)
        R = E @ R
        X = X @ Ei
        Xi = E @ Xi

        # Row Elimination
        b = -R.get_row(r) / d
        E = elim_col(n, r, b)
        Ei = elim_col(n, r, -b)
        R = R @ E
        Y = Ei @ Y
        Yi = Yi @ E

    return MatrixDecomposition(Xi, X, Yi, Y, R)

def triangular_inverse(R):
    # Compute invserse of triangular matrix using back-substitution
    n,m = R.shape
    assert n == m
    return la.solve_triangular(R, Matrix.id(n))

# https://stackoverflow.com/a/55737198/1209380
def perm_inv(permutation):
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
    return inv

def rd_qr(A: Matrix):
    A = Matrix(A)
    m, n = A.shape
    Q, R, p = la.qr(A, pivoting=True)
    Q = Matrix(Q)
    R = Matrix(R)
    # P = Matrix.perm(p)
    # assert (A @ P).is_sim(Q @ R)
    r = (np.abs(R.get_diag()) > EPS).sum()
    if r == 0:
        X = Matrix.id(m)
        Y = Matrix.id(n)
        return MatrixDecomposition(X, X, Y, Y, A)
    # R = [ R0 R1 ]
    #     [  0  0 ]
    R0 = R[:r, :r]
    R1 = R[:r, r:]
    R0i = triangular_inverse(R0)
    Su = Matrix(R0i).join_col(-R0i @ R1)
    Sl = Matrix.null(n - r, r).join_col(Matrix.id(n - r))
    S = Su.join_row(Sl)
    Siu = Matrix(R0).join_col(R1)
    Sil = Matrix.null(n - r, r).join_col(Matrix.id(n - r))
    Si = Siu.join_row(Sil)
    # assert (S @ Si).is_sim(Matrix.id(n))
    X = Q.T
    Xi = Q
    # Calculating products with permutation matrices takes more time than the QR decomposition
    # itself. Re-arranging the rows directly is much faster:
    Y = S.perm_row(perm_inv(p)) # = P @ S
    Yi = Si.perm_col(perm_inv(p)) # = Si @ P.T
    # Calculating the reduced diagonal matrix takes about half the time of the QR decomposition.
    # Since we know what the result should be we return it directly:
    # B = R @ S
    B = Matrix.null(m,n)
    np.fill_diagonal(B[:r,:r], 1)
    return MatrixDecomposition(X, Xi, Y, Yi, B)

def rd_svd(A: Matrix):
    A = Matrix(A)
    U, S, V = A.svd()  # A = U @ S @ V
    # assert(A.is_sim(U @ S @ V))
    return MatrixDecomposition(U.T, U, V.T, V, S)


def rank_decomposition(self, method="svd"):
    if method == "svd":
        return rd_svd(self)
    elif method == "qr":
        return rd_qr(self)
    elif method == "lu":
        return rd_lu(self)
    else:
        raise NotImplemented(f"Unknown method: {method}")
