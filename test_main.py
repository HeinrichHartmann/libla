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

def test_mf_gauss():
    def check(A):
        res = mf_gauss(A)
        assert res.is_valid()
        assert res.R.is_diagonal()
        assert (res.map(A) - res.R).is_null()

    for A in TEST_A:
        check(A)
