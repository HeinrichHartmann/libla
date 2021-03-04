import numpy as np
import scipy as sp
import scipy.linalg as la
from matplotlib import pyplot as plt

from dataclasses import dataclass

# import matplotlib.pyplot as pt

EPS = 10 ** -5


class mat(np.ndarray):
    # https://numpy.org/devdocs/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray
    def __new__(cls, X):
        obj = np.asarray(X).view(cls)
        return obj

    @classmethod
    def from_str(cls, s):
        return mat([[float(x) for x in r.split()] for r in s.strip().split("\n")])

    @classmethod
    def rand(cls, m, n=None):
        if not n:
            n = m
        x = np.random.normal(size=n * m)
        x.resize(m, n)
        return cls(x)

    @classmethod
    def id(cls, n):  # "identity (id)"
        return mat(np.identity(n))

    @classmethod
    def perm(cls, p):
        "permutation matrix, that maps e[i] -> e[P[i]]"
        return mat.id(len(p))[:, p]

    @classmethod
    def perm_tr(cls, n, i, j):
        p = np.arange(n)
        p[i] = j
        p[j] = i
        return cls.perm(p)

    @classmethod
    def perm_rot(cls, n, k):
        p = np.arange(n) + k
        p[p >= n] -= n
        return cls.perm(p)

    @classmethod
    def diag(cls, diag, m=None, n=None):
        "Square matrix with given diagnoal entries"
        if not m:
            m = len(diag)
        if not n:
            n = len(diag)
        X = np.zeros([m, n])
        for i, x in enumerate(diag):
            X[i, i] = x
        return X

    @classmethod
    def rand_rk(cls, n, r):
        "random nxn matrix of rk=r."
        return mat.rand(n, r) @ mat.rand(r, n)

    @classmethod
    def rand_rk_svd(cls, n, r):
        A = np.random.randn(n, n)
        U, sigma, VT = la.svd(A)
        sigma[r:] = 0
        return mat((U * sigma) @ VT)

    def is_null(self):
        return np.allclose(self, np.zeros(self.shape), atol=EPS)

    def is_diagonal(self):
        cp = self.copy()
        np.fill_diagonal(cp, 0)
        return cp.is_null()

    def inv(self):
        return mat(la.inv(self))

    def pinv(self):
        return mat(la.pinv(self, cond=EPS))

    def perm_col(self, p):
        return self[:, p]

    def perm_row(self, p):
        return self[p, :]

    def svd(self):
        U, s, V = linalg.svd(self)
        return (mat(U), mat(s), mat(V))

    def rank(self):
        s = linalg.svd(self, compute_uv=False)
        return len(s[s > EPS])

    def join_col(self, Y):
        return mat(np.concatenate([self, Y], axis=1))

    def join_row(self, Y):
        return mat(np.concatenate([self, Y], axis=0))

    def solve(self, b):
        "return solution of A x = b"
        return mat(la.solve(self, b))

    def lstsq(self, b):
        "return solution of | A x - b | minimal"
        x, res, rk, s = la.lstsq(X, X[:, 1])
        return la.lstsq(self, b)

    def row(self, i):
        "select i-th row"
        return self[i, :].to_row()

    def col(self, j):
        "select j-th col"
        return self[:, j].to_col()

    def to_row(self):
        return mat(self.reshape([1, len(self)]))

    def to_col(self):
        return mat(self.reshape([len(self), 1]))

    def set_row(self, i, r):
        self[i, :] = r[0, :]

    def set_col(self, j, c):
        self[:, j] = c[:, 0]

    def vsum(self, B):
        """return block diagonal matrix with entries A,B"""
        A = self
        m, n = A.shape
        p, q = B.shape
        X = np.zeros([m + p, n + q])
        X[:m, :n] = A
        X[m:, n:] = B
        return mat(X)

    def _repr_latex_(self):
        """Returns a LaTeX bmatrix
        :a: numpy array
        :returns: LaTeX bmatrix as a string
        """
        if len(self.shape) > 2:
            raise ValueError("bmatrix can at most display two dimensions")

        def fmt(x):
            if x == 0:
                return "."
            if np.abs(x) < EPS:
                return "0."
            return "{:.2g}".format(x)

        temp_string = np.array2string(
            self,
            formatter={"float_kind": fmt},
            edgeitems=3,
        )
        lines = temp_string.replace("[", "").replace("]", "").splitlines()
        rv = [r"\begin{bmatrix}"]
        rv += ["  " + " & ".join(l.split()) + r"\\" for l in lines]
        rv += [r"\end{bmatrix}"]
        return "\n".join(rv)

    def __str__(self):
        def fmt(x):
            if x == 0:
                return "     ."
            if np.abs(x) < EPS:
                return "    0."
            return "{:6.2g}".format(x)

        temp_string = np.array2string(
            self,
            formatter={"float_kind": fmt},
            edgeitems=3,
        )
        m,n = self.shape
        return f"mat[{m},{n}]"+ "{\n" + temp_string.replace("[[", " [").replace("]]", "] ") + "\n}"

    def plot(self):
        plt.imshow(np.log10(1e-15 + np.abs(self)))


class Vect(mat):
    "Vector sub-spaces -- represented as orthogonal systems"

    @classmethod
    def Im(cls, X):
        "Returns ONB of Im(self)"
        Q, R, p = la.qr(X, pivoting=True)  # A @ P = Q @ R.
        D = np.diag(R)
        r = int((np.abs(D) > EPS).sum())  # rank
        return Vect(Q[:, :r])

    @classmethod
    def Ker(cls, X):
        "Returns ONB of Ker(self)"
        # Ker(A) = Im(A^t)^perp
        Q, R, p = la.qr(X.T, pivoting=True)  # A @ P = Q @ R
        D = np.diag(R)
        r = int((np.abs(D) > EPS).sum())  # rank
        return Vect(Q[:, r:])  # take columns not touched by R

    def dim(self):
        return self.shape[1]

    def perp(self):
        "orthogonal complement"
        return Vect.Ker(self.T)  # Ker(X.T) = Im(X)^perp

    def contains(self, Y):
        return (self @ self.T @ Y - Y).is_null()

    def project(self, Y):
        "project columns of Y into self"
        return self @ self.T @ Y

    def add(self, Y):
        return Vect.Im(self.join_col(Y))


def perm_transposition(n, a, b):
    p = np.arange(n)
    p[a] = b
    p[b] = a
    return p

@dataclass
class MatrixFactorizationResult:
    A: mat
    X: mat
    Xi: mat
    Y: mat
    Yi: mat

    def is_valid(self):
        m,n = A.shape
        return (mat.id(m) - X @ Xi).is_null() \
            and (mat.id(n) - Y @ Yi).is_null() \
            and (R - Xi @ A @ Y).is_null() \
            (A - X @ R @ Yi).is_null()

    def plot(self):
        fig, (ax1,ax2,ax3) = plt.subplots(
            nrows=1,ncols=3,figsize=(15,5))
        def prep(A): return np.log10(1e-15 + np.abs(A))
        ax1.imshow(prep(self.X))
        ax2.imshow(prep(self.A))
        ax3.imshow(prep(self.Yi))

    def __iter__(self):
        # This is so that
        #   A,X,Xi,Y,Yi = factor(...)
        # and
        #   res = MatrixFactorizationResult(factor())
        # work
        return iter([self.A, self.X, self.Xi, self.Y, self.Yi])


def elim2_row(n, r, b_orig):
    b = b_orig.copy()
    """returns nxn Gauss elimination matrix"""
    E = mat.id(n)
    b[:r,:] = 0
    E.set_col(r, b)
    E[r, r] = 1
    return E


def elim2_col(n, r, b_orig):
    """returns nxn Gauss elimination matrix"""
    b = b_orig.copy()
    E = mat.id(n)
    b[:,:r] = 0
    E.set_row(r, b)
    E[r, r] = 1
    return E

def rd_gauss2(A: np.ndarray):
    """
    Returns MatrixFactorization with:
    X = product of permutation and lower triagonal matrix
    Y = product of permutation and upper triagonal matrix
    R = diagonal matrix
    """
    A = mat(A)
    m, n = A.shape

    X = mat.id(m)
    Xi = mat.id(m)
    R = A
    Y = mat.id(n)
    Yi = mat.id(n)

    for r in range(min(m,n)):
        # Pivot selection
        M = np.abs(R[r:,r:])
        pivot = np.unravel_index(np.argmax(M, axis=None), M.shape)
        pivot = [ pivot[0] + r, pivot[1] + r ]
        d = R[pivot[0], pivot[1]]
        if abs(d) < EPS:
            break
        P = mat.perm(perm_transposition(m, r, pivot[0]))
        Q = mat.perm(perm_transposition(n, r, pivot[1]))
        R = P @ R @ Q
        X = X @ P.T
        Xi = P @ Xi
        Yi = Q.T @ Yi
        Y = Y @ Q

        # Column elimination
        b = -R.col(r) / d
        E = elim2_row(m, r, b)
        Ei = elim2_row(m, r, -b)
        R = E @ R
        X = X @ Ei
        Xi = E @ Xi

        # Row Elimination
        b = -R.row(r) / d
        E = elim2_col(n, r, b)
        Ei = elim2_col(n, r, -b)
        R = R @ E
        Yi = Ei @ Yi
        Y = Y @ E

    return MatrixFactorizationResult(R, X, Xi, Y, Yi)
