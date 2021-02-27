import numpy as np
import scipy as sp
import scipy.linalg as la
# import matplotlib.pyplot as pt

EPS = 10**-5

class mat(np.ndarray):
    # https://numpy.org/devdocs/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray
    def __new__(cls, X):
        obj = np.asarray(X).view(cls)
        return obj

    def from_str(s):
        return mat(
         [[float(x) for x in r.split()] for r in s.strip().split("\n")]
        )
    
    @classmethod
    def col(cls, X):
        return cls(np.asarray(X).reshape([len(X), 1]))

    @classmethod
    def row(cls, X):
        return cls(np.asarray(X).reshape([1, len(X)]))
    
    @classmethod
    def rand(cls, m, n = None):
        if not n:
            n = m
        x = np.random.normal(size=n*m)
        x.resize(m,n)
        return cls(x)
    
    @classmethod
    def id(cls, n): # "identity (id)"
        return mat(np.identity(n))
    
    @classmethod
    def perm(cls, p):
        "permutation matrix, that maps e[i] -> e[P[i]]"
        return mat.id(len(p))[:,p]
    
    @classmethod
    def perm_tr(cls, n, i, j):
        p = np.arange(n)
        p[i] = j; p[j] = i
        return cls.perm(p)

    @classmethod
    def perm_rot(cls, n, k):
        p = np.arange(n) + k
        p[p >= n] -= n
        return cls.perm(p)
    
    @classmethod
    def diag(cls, diag, m=None, n=None):
        "Square matrix with given diagnoal entries"
        if not m: m = len(diag)
        if not n: n = len(diag)
        X = np.zeros([m, n])
        for i, x in enumerate(diag):
            X[i,i] = x
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
        return self[:,p]
    
    def perm_row(self, p):
        return self[p,:]

    def svd(self):
        U,s,V = linalg.svd(self)
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
        x, res, rk, s = la.lstsq(X, X[:,1])
        return la.lstsq(self, b)
    
    def _repr_latex_(self):
        """Returns a LaTeX bmatrix
        :a: numpy array
        :returns: LaTeX bmatrix as a string
        """
        if len(self.shape) > 2:
            raise ValueError('bmatrix can at most display two dimensions')
        def fmt(x):
            if x == 0: return "."
            if np.abs(x) < EPS: return "0."
            return "{:.2g}".format(x)
        temp_string = np.array2string(
            self, 
            formatter={'float_kind': fmt },
            edgeitems = 3,
        )
        lines = temp_string.replace('[', '').replace(']', '').splitlines()
        rv = [r'\begin{bmatrix}']
        rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
        rv +=  [r'\end{bmatrix}']
        return '\n'.join(rv)
    
    def plot(self):
        pt.imshow(np.log10(1e-15+np.abs(self)))
        # pt.colorbar()


class Vect(mat):
    "Vector sub-spaces -- represented as orthogonal systems"

    @classmethod
    def Im(cls, X):
        "Returns ONB of Im(self)"
        Q,R,p = la.qr(X, pivoting=True) # A @ P = Q @ R.
        D = np.diag(R)
        r = int((np.abs(D) > EPS).sum()) # rank
        return Vect(Q[:,:r])
    
    @classmethod
    def Ker(cls, X):
        "Returns ONB of Ker(self)"
        # Ker(A) = Im(A^t)^perp
        Q,R,p = la.qr(X.T, pivoting=True) # A @ P = Q @ R
        D = np.diag(R)
        r = int((np.abs(D) > EPS).sum()) # rank
        return Vect(Q[:,r:]) # take columns not touched by R

    def dim(self):
        return self.shape[1]
    
    def perp(self):
        "orthogonal complement"
        return Vect.Ker(self.T) # Ker(X.T) = Im(X)^perp
        
    def contains(self, Y):
        return (self @ self.T @ Y - Y).is_null()
    
    def project(self, Y):
        "project columns of Y into self"
        return self @ self.T @ Y
    
    def add(self, Y):
        return Vect.Im(self.join_col(Y))

def ldu(A : np.ndarray):
    """
    returns P,L,D,U,Q = A
    """
    n,m = A.shape
    if n == 0 or m == 0:
        P = mat.id(n)
        L = mat.id(n)
        D = A
        U = mat.id(m)
        Q = mat.id(m)
        return (P,L,D,U,Q)
