import pytest
from main import *
import numpy as np


TEST_A = [
    mat.from_str("""0 0 0"""),
    mat.from_str("""
    1 1 1
    3 1 5
    """),
    mat.from_str("""
    0 1
    1 1
    """),
    mat.rand_rk(5,3),
    mat.rand_rk(30,3),
]

def test_ldu():
    def check(A):
        P,L,D,U,Q = ldu(A)
        assert (A - P @ L @ D @ U @ Q).is_null()
    for A in TEST_A:
        check(A)

def test_rd():
    def check(A):
        R,X,Xi,Y,Yi = rd_gauss(A)
        m,n = A.shape
        assert (mat.id(m) - X @ Xi).is_null()
        assert (mat.id(n) - Y @ Yi).is_null()
        assert (R - Xi @ A @ Y).is_null()
        assert (A - X @ R @ Yi).is_null()
        np.fill_diagonal(R, 0)
        assert R.is_null()
    for A in TEST_A:
        check(A)

def test_rd2():
    def check(A):
        R,X,Xi,Y,Yi = rd_gauss2(A)
        m,n = A.shape
        assert (mat.id(m) - X @ Xi).is_null()
        assert (mat.id(n) - Y @ Yi).is_null()
        assert (R - Xi @ A @ Y).is_null()
        assert (A - X @ R @ Yi).is_null()
        np.fill_diagonal(R, 0)
        assert R.is_null()
    for A in TEST_A:
        check(A)
