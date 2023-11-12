# tensor

An infinite dimensional tensor package.
```python
>>> from tensor import Tensor
>>> t = Tensor(((1, 2), (3, 4)))
>>> t
Tensor[[1 2]
       [3 4]]
>>> s = Tensor(((1,), (2,)))
>>> s
Tensor[[1]
       [2]]
>>> t + s
Tensor[[2 2]
       [5 4]]
```

## Installation

```console
pip install git+https://github.com/goessl/tensor.git
```

## Usage

This package provides a single class, `Tensor`, to handle infinite dimensional tensors.
A tensor can be initialized in three ways:
 - With the constructor `Tensor(coef)` that takes an array like object (something that is accepted by `numpy.asarray`) and wraps it into a tensor.
 - With the basis factors `Tensor.basis_tensor(d0, d1, ...)` that returns a tensor full of zeros and a one in the specified position.
 - With the random factory `Tensor.random(d0, d1, ...)` for a random tensor of the given dimensionality.
The objects are immutable (coefficients are internally stored in a write protected `numpy.array`) and zero-indexed.
```python
>>> from vector import Vector
>>> v = Vector((1, 2, 3))
>>> w = Vector.random(3)
>>> v
Vector(1, 2, 3, 0, ...)
>>> w
Vector(-0.5613820142699765, -0.028308921297709365, 0.8270724508948077, 0, ...)
>>> Vector.basis_tuple(3)
(0, 0, 0, 1)
```

Tensor object have
- a rank: `t.rank` &
- a shape: `t.shape`.

Container and sequence interfaces are implemented so the coefficients can be
- accessed by indexing: `t[2, 3]` (coefficients not set return to 0),
- casted to an array (implements `__array__`),
- compared: `t == s` (tuple of coefficients get compared),
- shifted: `t >> (3, 4), t << (1, 2)`,
- trimmed: `t.trim()` (trailing non-zero coefficients get removed) &
- rounded: `t.round(2)`.

Hilbert space operations are provided:
- Vector addition & subtraction `t + s, t - s` &
- scalar multiplication & division `2 * t, t / 2`.
The multiplicative operations are overloaded to perform scalar multiplication/division if the other argument is a scalar, or elementwise multiplication/division if both operands are `Tensors`s, `t * s`.

## Design decisions

The `Tensor` class not only wrapps a `numpy.array`, but also tries as best as
possible to use the standard built-in Python types (`int`, `float`, ...).
This makes calculations slower, but keeps arithmetic Python features
like unbounded integers.

## License (MIT)

Copyright (c) 2023 Sebastian Gössl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
