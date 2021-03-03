import pytest
from main import *

def test_ldu():
    A = mat.from_str("""0 0 0""")
    P,L,D,U,Q = ldu(A)
    assert (A - P @ L @ D @ U @ Q).is_null()
    
    A = mat.from_str("""
    1 1 1
    3 1 5
    """)
    P,L,D,U,Q = ldu(A)
    print(P,L,D,U,Q)
    assert (A - P @ L @ D @ U @ Q).is_null()

    A = mat.from_str("""
    0 1
    1 1
    """)
    P,L,D,U,Q = ldu(A)
    assert (A - P @ L @ D @ U @ Q).is_null()

    A = mat.rand_rk(5,3)
    P,L,D,U,Q = ldu(A)
    assert (A - P @ L @ D @ U @ Q).is_null()


def test_rd():
    A = mat.from_str("""0 0 0""")
    R,X,Xi,Y,Yi = rd_gauss(A)
    print(A, X, R, Yi)

    A = mat.from_str("""
    1 1 1
    3 1 5
    """)
    R,X,Xi,Y,Yi = rd_gauss(A)
    print(A, X, R, Yi)

    A = mat.from_str("""
    0 1
    1 1
    """)
    R,X,Xi,Y,Yi = rd_gauss(A)
    print(A, X, R, Yi)

    A = mat.rand_rk(5,3)
    R,X,Xi,Y,Yi = rd_gauss(A)
    print(A, X, R, Yi)

    A = mat.rand_rk(30,3)
    R,X,Xi,Y,Yi = rd_gauss(A)
    print(A, X, R, Yi)
