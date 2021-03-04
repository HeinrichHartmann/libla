import pytest
from main import *
import numpy as np

from hypothesis import given, example, assume, strategies as st
import hypothesis.extra.numpy as hnp

def test_Matrix():
    M = Matrix.id(3)
    assert M.rank() == 3
    assert M.is_sim(M.inv())
    assert M.is_sim(M.pinv())
    assert M.is_diagonal()
    assert not M.is_null()

    X = Matrix.from_str("1 1 0; 0 1 0; 0 0 0")
    Y = Matrix.from_str("1 -1 0; 0 1 0; 0 0 0")
    assert X.rank() == 2
    assert X.pinv().is_sim(Y)

    Z = Matrix.from_str("1 -1 0; 0 1 0; 0 0 0; 0 0 0")
    W = Matrix.from_str("1 -1 0 1; 0 1 0 1; 0 0 0 1")
    assert Y.join_row(Matrix([0,0,0]).to_row()).is_sim(Z)
    assert Y.join_col(Matrix([1,1,1]).to_col()).is_sim(W)

@given(hnp.arrays(np.int32, (3,3), elements=st.sampled_from([0,1])))
def test_rand_bin_matrix(A):
    M = Matrix(A)
    P = M @ M.pinv()
    Q = M.pinv() @ M
    assert P.is_sim(P @ P)
    assert Q.is_sim(Q @ Q)

@given(hnp.arrays(np.int64, (3,3)))
def test_rand_int_matrix(A):
    M = Matrix(A)
    P = M @ M.pinv()
    Q = M.pinv() @ M
    assert P.is_sim(P @ P)
    assert Q.is_sim(Q @ Q)

@given(hnp.arrays(np.float64, (3,3)))
def test_rand_float_matrix(A):
    assume(np.isfinite(A).all())
    M = Matrix(A)
    P = M @ M.pinv()
    Q = M.pinv() @ M
    assert P.is_sim(P @ P)
    assert Q.is_sim(Q @ Q)

@given(hnp.arrays(np.float64, (3,3), elements=st.floats(0,1)))
@example(Matrix.from_str("0 0 0"))
@example(Matrix.from_str("0 0 0"))
@example(Matrix.from_str("1 1 1 ; 3 1 5"))
@example(Matrix.from_str("0 1 ; 1 1"))
@example(Matrix.rand_rk(5,3))
@example(Matrix.rand_rk(30,3))
def test_mf_gauss(A):
    assume(np.isfinite(A).all())
    res = md_gauss(A)
    assert res.is_valid()
    assert res.R.is_diagonal()
    assert (res.map(A) - res.R).is_null()
