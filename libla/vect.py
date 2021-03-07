import numpy as np

from . import Matrix


class Vect(Matrix):
    "Vector sub-spaces represented by a matrix"

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
