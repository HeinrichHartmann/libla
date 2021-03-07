# libla - A Linear Algebra Library for Python/numpy

In numeric applications, all Linear Algebra concepts are conflated to a single object: The numpy array.
In this library we provide a collection of sub-classes of np.array that specialize the interpretation of the
array as linear map (Matrix), sub-vector space (Vect), etc. and provide a collection of common high-level
operations for the concept at hand.

**Disclaimer** This library is in it's infancy. We do have quite some tests, so correctness should be OK, but expect things to change without warning.

## Example

```python
> X = Matrix.rand_rk(10, 5, 3) # random 10x5 matrix of rank 3

> X.rank()
3

> X # pretty printing by default
Matrix[10,5]{
 [     1  -0.62  -0.16   -1.1   -2.6]
 [  0.58  0.019    1.6    1.8     -1]
 [ -0.36  -0.54  -0.35     -2  0.018]
 [ -0.17 -0.078   -1.5   -1.5   0.21]
 [ -0.68  -0.42    1.4   -0.4   0.75]
 [-0.054 -0.019   0.76   0.54   0.08]
 [  0.44   -0.5  0.032     -1   -1.4]
 [ -0.36  -0.65      1   -1.2  -0.11]
 [   0.2  -0.01  0.051   0.15  -0.37]
 [  0.74   0.21   -1.6  -0.22   -1.1]
}

> X[1:3] # numpy operations keep working and yield Matrix objects
Matrix[2,5]{
 [  0.58  0.019    1.6    1.8     -1]
 [ -0.36  -0.54  -0.35     -2  0.018] 
}

> V = Vect.Im(X) # Image vector space
> V.dim()
3

> V.contains(X.get_col(3))
True

> V.project([1,2,3,4,5,6,7,8,9,10]).to_row()
Matrix[1,10]{
 [   6.9   0.25    3.6   0.94    1.2  -0.32    4.7    3.8   0.45   0.62]
}
```

See ./examples for more usage examples.

## Applications

* https://www.heinrichhartmann.com/posts/2021-03-08-rank-decomposition/

## Installation

```
pip install libla
```

## Changelog

* 2020-03-07 v0.2 Revised project structure
