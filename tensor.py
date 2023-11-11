import numpy as np
import operator



class Tensor:
    """An infinite dimensional tensor class."""
    
    #construction stuff
    def __init__(self, coef):
        """Creates a new tensor with the given coefficients."""
        #cant differentiate between a tuple as a rank one tensor
        #or a tuple as a "basis tensor" request
        self.coef = np.asarray(coef, dtype=object)
        self.coef.flags.writeable = False
    
    @staticmethod
    def basis_tensor(*i):
        #see Tensor.random
        t = np.zeros(np.add(i, 1), dtype=int)
        t[i] = 1
        return Tensor(t)
    
    @staticmethod
    def random(*n):
        """Creates a tensor of the given dimensionality
        with normal distributed coefficients."""
        #careful, needs astype(float).astype(object)
        #but the latter happens in the constructor
        return Tensor(np.random.normal(size=n).astype(float))
    
    
    
    #container stuff
    def __getitem__(self, key):
        try:
            return self.coef[key]
        except IndexError:
            return 0
    
    def __array__(self):
        return self.coef
    
    def __eq__(self, other):
        return np.all(self.coef == other.coef)
    
    def __lshift__(self, other):
        return type(self)(self[tuple(slice(o, None) for o in other)])
    
    @staticmethod
    def _pad(arr, pad_width):
        """numpy.pad but with native integers, as numpy.pad always uses numpy.int64."""
        for a, p in enumerate(pad_width):
            before, after = list(arr.shape), list(arr.shape)
            before[a], after[a] = p[0], p[1]
            arr = np.concatenate((np.zeros(before, dtype=int),
                                  arr,
                                  np.zeros(after, dtype=int)), axis=a)
        return arr
    
    def __rshift__(self, other):
        #return type(self)(np.pad(s.coef, tuple((o, 0) for o in other)))
        #see Tensor._pad
        return type(self)(Tensor._pad(self.coef, tuple((o, 0) for o in other)))
    
    def trim(self):
        """Removes all trailing near zero coefficients."""
        #https://stackoverflow.com/a/65547931/7367030
        #np.isclose doesn't accept object arrays
        #math.isclose doesn't work near 0 (with default abs=0)
        idcs = np.where(np.invert(np.vectorize(np.isclose)(self.coef, 0)))
        slices = tuple(slice(0, idx.max()+1) for idx in idcs)
        return type(self)(self[slices])
    
    def round(self, ndigits=None):
        """Rounds all coefficients to the given precision."""
        #numpy.round doesn't accept object arrays
        f = np.vectorize(lambda x: round(x, ndigits))
        return type(self)(f(self.coef)).trim()
    
    
    
    #Hilbert space stuff
    @property
    def rank(self):
        return self.coef.ndim
    
    @property
    def shape(self):
        return self.coef.shape
    
    
    
    #vector space operations
    #The return type of the arithmetic operations is determined by
    #the first argument (self) to enable correctly typed results for subclasses.
    #E.g. the sum of two MultiPolys should again be a MultiPoly,
    #and not a Tensor.
    @staticmethod
    def map_zip(f, s, t):
        """Applies f(v, w) elementwise if possible,
        otherwise elementwise in the first argument."""
        try:
            shape = np.minimum(s.coef.shape, t.coef.shape)
            r = f(s[tuple(slice(0, s) for s in shape)],
                  t[tuple(slice(0, s) for s in shape)])
        except AttributeError:
            r = f(s.coef, t)
        return type(s)(r)
    
    @staticmethod
    def map_zip_longest(f, s, t):
        """Applies f(s, t) elementwise if possible,
        otherwise elementwise in the first argument."""
        try:
            shape = np.maximum(s.coef.shape, t.coef.shape)
            r = f(Tensor._pad(s.coef, tuple((0, s-l) for s, l in zip(shape, s.coef.shape))),
                  Tensor._pad(t.coef, tuple((0, s-l) for s, l in zip(shape, t.coef.shape))))
        except AttributeError:
            r = f(s.coef, t)
        return type(s)(r)
    
    #implement vector space operations like they would be correct on paper:
    #s+t, s-t, at, ta, t/a
    def __add__(self, other):
        return Tensor.map_zip_longest(operator.add, self, other)
    
    def __sub__(self, other):
        return Tensor.map_zip_longest(operator.sub, self, other)
    
    def __mul__(self, other):
        return Tensor.map_zip(operator.mul, self, other)
    __rmul__ = __mul__
    
    def __truediv__(self, other):
        return Tensor.map_zip(operator.truediv, self, other)
    
    
    
    #python stuff
    def __str__(self):
        return 'Tensor' + np.array2string(self.coef, prefix='Tensor')
