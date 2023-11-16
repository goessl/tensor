from tensor import Tensor



assert isinstance(Tensor((1,))[0], int)
assert Tensor.basis_tensor(1, 2) == Tensor(((0, 0, 0),
                                            (0, 0, 1)))
assert isinstance(Tensor.basis_tensor(1, 2)[0, 0], int)
assert isinstance(Tensor.random(1)[0], float)

t = Tensor(((1, 2),
            (3, 4),
            (5, 6)))

assert t[2, 1] == 6
assert isinstance(t[2, 1], int)
assert t[999, 238974] == 0
assert isinstance(t[999, 238974], int)
assert t<<(2, 1) == Tensor(((6,),))
assert isinstance((t<<(2, 1))[0, 0], int)
assert t>>(1, 2) == Tensor(((0, 0, 0, 0),
                            (0, 0, 1, 2),
                            (0, 0, 3, 4),
                            (0, 0, 5, 6)))
assert isinstance((t>>(1, 2))[0, 0], int)
assert Tensor((1, 0)).trim() == Tensor((1,))# \
#        and Tensor(tuple()).trim() == Tensor(tuple())
assert isinstance(Tensor((1, 0)).trim()[0, 0], int)

s = Tensor(((7,  8),
            (9, 10)))
assert t+s == Tensor((( 8, 10),
                        (12, 14),
                        ( 5,  6)))
assert isinstance((t+s)[0, 0], int)
assert t-s == Tensor(((-6, -6),
                        (-6, -6),
                        ( 5,  6)))
assert isinstance((t-s)[0, 0], int)
assert 2*t == t*2 == Tensor((( 2,  4),
                                ( 6,  8),
                                (10, 12)))
assert isinstance((2*t)[0, 0], int)
assert t/2 == Tensor(((0.5, 1),
                        (1.5, 2),
                        (2.5, 3)))
assert isinstance((t/2)[0, 0], float) and isinstance((t/2)[0, 1], float)
