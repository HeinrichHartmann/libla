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
    print(A)
    print(P,L,D,U,Q)
    assert (A - P @ L @ D @ U @ Q).is_null()

    A = mat.from_str("""
    0 1
    1 1
    """)
    P,L,D,U,Q = ldu(A)
    print(P,L,D,U,Q)
    assert (A - P @ L @ D @ U @ Q).is_null()

    A = mat.rand_rk(5,3)
    P,L,D,U,Q = ldu(A)
    print(P,L,D,U,Q)
    assert (A - P @ L @ D @ U @ Q).is_null()

