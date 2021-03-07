import numpy as np
import scipy as sp
import scipy.linalg as la
from matplotlib import pyplot as plt
from dataclasses import dataclass

from . import EPS

print(EPS)

# Forward declaration
Matrix = None
rank_decomposition = None

@dataclass
class MatrixTransform:
    """
    Bi-variant matrix transformation:

    A --map--> X A Y = R residual matrix
    R --imap--> Xi R Yi = A original matrix
    """

    X: Matrix
    Xi: Matrix
    Y: Matrix
    Yi: Matrix

    def is_valid(self):
        X, Xi, Y, Yi = MatrixTransform.__iter__(self)
        m = X.shape[0]
        n = Y.shape[0]
        return (Matrix.id(m) - X @ Xi).is_null() and (Matrix.id(n) - Y @ Yi).is_null()

    def map(self, v):
        return self.X @ v @ self.Y

    def imap(self, w):
        return self.Xi @ w @ self.Yi

    def __iter__(self):
        # convert to tuple
        return iter([self.X, self.Xi, self.Y, self.Yi])


@dataclass
class MatrixDecomposition(MatrixTransform):
    """Holds results of matrix decomposition: R = X A Y"""

    R: Matrix

    def __iter__(self):
        # convert to tuple
        return iter([self.R, self.X, self.Xi, self.Y, self.Yi])

    def plot(self):
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

        def prep(A):
            return np.log10(1e-15 + np.abs(A))

        ax1.imshow(prep(self.X))
        ax2.imshow(prep(self.R))
        ax3.imshow(prep(self.Y))


class Matrix(np.ndarray):
    # https://numpy.org/devdocs/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray
    def __new__(cls, X):
        obj = np.asarray(X).view(cls)
        return obj

    @classmethod
    def from_str(cls, s):
        return Matrix(
            [
                [float(x) for x in r.split()]
                for r in s.strip().replace("\n", ";").split(";")
            ]
        )

    @classmethod
    def rand(cls, m, n=None):
        """Create a mxn matrix with random entries"""
        if not n:
            n = m
        x = np.random.normal(size=n * m)
        x.resize(m, n)
        return cls(x)

    @classmethod
    def rand_rk(cls, m, n, r):
        A = np.random.randn(m, n)
        U, sigma, VT = la.svd(A)
        sigma[r:] = 0
        S = np.zeros(A.shape)
        np.fill_diagonal(S, sigma)
        return Matrix(U @ S @ VT)

    @classmethod
    def id(cls, n):  # "identity (id)"
        return Matrix(np.identity(n))

    @classmethod
    def null(cls, m, n):
        return Matrix(np.zeros([m, n]))

    @classmethod
    def perm(cls, p):
        "permutation matrix, that maps e[i] -> e[P[i]]"
        return Matrix.id(len(p))[:, p]

    @classmethod
    def perm_tr(cls, n, i, j):
        "Return permutation matrix"
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
        "Create square matrix with given diagnoal entries"
        if not m:
            m = len(diag)
        if not n:
            n = len(diag)
        X = np.zeros([m, n])
        for i, x in enumerate(diag):
            X[i, i] = x
        return X

    def is_null(self):
        return np.allclose(self, np.zeros(self.shape), atol=EPS)

    def is_diagonal(self):
        cp = self.copy()
        np.fill_diagonal(cp, 0)
        return cp.is_null()

    def is_sim(self, B):
        return (self.shape == B.shape) and (self - B).is_null()

    def inv(self):
        "Return Inverse Matrix"
        m, n = self.shape
        if n == 0 and m == 0:
            return self  # 0x0 matrix is self inverse
        return Matrix(la.inv(self))

    def pinv(self):
        "Return pseudo inverse Matrix"
        return Matrix(la.pinv(self, cond=EPS))

    def svd(self):
        """Returns U,S,Vh so that self = U @ S @ Vh"""
        U, s, Vh = la.svd(self)
        S = np.zeros(self.shape)
        np.fill_diagonal(S, s)
        return (Matrix(U), Matrix(S), Matrix(Vh))

    def rank(self):
        s = la.svd(self, compute_uv=False)
        return (s > EPS).sum()

    def join_col(self, Y):
        return Matrix(np.concatenate([self, Y], axis=1))

    def join_row(self, Y):
        return Matrix(np.concatenate([self, Y], axis=0))

    def solve(self, b):
        "return solution of A x = b"
        return Matrix(la.solve(self, b))

    def lstsq(self, b):
        "return solution of | A x - b | minimal"
        x, res, rk, s = la.lstsq(X, X[:, 1])
        return la.lstsq(self, b)

    def get_row(self, i):
        "select i-th row"
        return self[i, :].to_row()

    def get_col(self, j):
        "select j-th col"
        return self[:, j].to_col()

    def get_diag(self):
        return np.array(np.diag(self))

    def to_row(self):
        return Matrix(self.reshape([1, self.size]))

    def to_col(self):
        return Matrix(self.reshape([self.size, 1]))

    def set_row(self, i, r):
        self[i, :] = Matrix(r).to_row()[0, :]

    def set_col(self, j, c):
        self[:, j] = Matrix(c).to_col()[:, 0]

    def rank_decomposition(self, *args, **kwargs):
        return rank_decomposition(self, *args, **kwargs)

    def vsum(self, B):
        """return block diagonal matrix with entries A,B"""
        A = self
        m, n = A.shape
        p, q = B.shape
        X = np.zeros([m + p, n + q])
        X[:m, :n] = A
        X[m:, n:] = B
        return Matrix(X)

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
        m, n = self.shape
        return (
            f"mat[{m},{n}]"
            + "{\n"
            + temp_string.replace("[[", " [").replace("]]", "] ")
            + "\n}"
        )

    def plot(self):
        plt.imshow(np.log10(1e-15 + np.abs(self)))

#
# Late import of rank_decomposition to avoid circular imports
#
# We need the Matrix class to be fully defined, before we can import this module
from libla.rank_decomposition import rank_decomposition
