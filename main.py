import numpy as np
import scipy as sp
import scipy.linalg as la

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

    def is_null(self):
        return np.allclose(self, np.zeros(self.shape), atol=EPS)

    def solve(self, b):
        "return solution of A x = b"
        return mat(la.solve(self, b))

    def lstsq(self, b):
        "return solution of | A x - b | minimal"
        x, res, rk, s = la.lstsq(X, X[:, 1])
        return la.lstsq(self, b)

    def row(self, i):
        "select i-th row"
        return self[i,:].to_row()

    def col(self, j):
        "select j-th col"
        return self[:,j].to_col()

    def to_row(self):
        return mat(self.reshape([1, len(self)]))

    def to_col(self):
        return mat(self.reshape([len(self), 1]))

    def set_row(self, i, r):
        self[i,:] = r[0,:]

    def set_col(self, j, c):
        self[:,j] = c[:,0]

    def vsum(self, B):
        """return block diagonal matrix with entries A,B"""
        A = self
        m,n = A.shape
        p,q = B.shape
        X = np.zeros([m+p, n+q])
        X[:m,:n] = A
        X[m:,n:] = B
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
        return "mat{\n" + temp_string.replace("[[", " [").replace("]]", "] ") + "\n}"


    def plot(self):
        pt.imshow(np.log10(1e-15 + np.abs(self)))
        # pt.colorbar()



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

def elim(n,b):
    """returns nxn Gauss elimination matrix"""
    E = mat.id(n)
    E.set_col(0, b)
    E[0,0] = 1
    return E

def elim_col(n,b):
    """returns nxn Gauss elimination matrix"""
    E = mat.id(n)
    E.set_row(0, b)
    E[0,0] = 1
    return E

def ldu(A: np.ndarray):
    """
    returns P,L,D,U,Q = A
    """
    m, n = A.shape
    A = mat(A)

    # TODO: Implement LDU by mutating those matrices
    P = mat.id(m)
    L = mat.id(m)
    D = A
    U = mat.id(n)
    Q = mat.id(n)

    # 1. Termination Conditions
    if n == 0 or m == 0 or D.is_null():
        return (P, L, D, U, Q)

    # 2. Pivot Selection
    pivot = np.unravel_index(np.argmax(D, axis=None), D.shape)
    d = D[pivot]
    P = mat.perm(perm_transposition(m, 0, pivot[0]))
    Q = mat.perm(perm_transposition(n, 0, pivot[1]))
    D = P @ A @ Q
    assert D[0, 0] == d
    assert (A - P @ L @ D @ U @ Q).is_null()

    # 3. Column elimination
    b = D.col(0) / d
    D = elim(m, -b) @ D
    L = elim(m, b)
    assert (A - P @ L @ D @ U @ Q).is_null()

    # 3. Row elimination
    b= D.row(0) / d
    D = D @ elim_col(n,-b)
    U = elim_col(n, b)
    assert (A - P @ L @ D @ U @ Q).is_null()

    # 4. Recursion
    B = D[1:,1:]
    P2, L2, D2, U2, Q2 = ldu(B)
    assert (B - P2 @ L2 @ D2 @ U2 @ Q2).is_null()
    P3 = mat.id(1).vsum(P2)
    L3 = mat.id(1).vsum(L2)
    D3 = (mat.id(1) * d).vsum(D2)
    U3 = mat.id(1).vsum(U2)
    Q3 = mat.id(1).vsum(Q2)
    P = P @ P3
    L = P3.T @ L @ P3 @ L3 # This is again lower triangular!
    D = D3
    U = U3 @ Q3 @ U @ Q3.T # This is again upper triangular!
    Q = Q3 @ Q
    assert (A - P @ L @ D @ U @ Q).is_null()
    return (P, L, D, U, Q)
