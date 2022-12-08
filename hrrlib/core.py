from __future__ import annotations
import numpy as np
from hrrlib.errors import PatternMatchError, DimensionsNotEqual


class PatternValidator:
    REAL_KINDS = ['i', 'f']
    COMPLEX_KINDS = ['c']

    def __init__(
        self, 
        data: np.ndarray, 
        tol: float = 1e-8, 
        reduced: bool = False, 
        even=False
    ):
        self._data = data
        self._kind = data.dtype.kind
        self._tol = tol
        self._reduced = reduced
        self._even = even

    @property
    def is_valid(self):  
        # TODO: Make all the falses raise errors for better clarity of issues
        if self.kind in self.REAL_KINDS:
            return True
        elif self.kind in self.COMPLEX_KINDS:
            # Check if legal (conjugate symmetric) pattern
            data = self._data
            n = len(data)
            if self._reduced:
                if self._even:
                    # Check if first and last digit are real (imag ~= 0)
                    if not (np.abs(data.imag[0]) <= self._tol and np.abs(data.imag[-1]) <= self._tol):
                        return False
                else:
                    # Check if first digit is real (imag ~= 0)
                    if not (np.abs(data.imag[0]) <= self._tol):
                        return False
                return True
            else:
                if not n % 2:  # if even
                    # Check if first and middle digit are real (imag ~= 0)
                    if not (np.abs(data.imag[0]) <= self._tol and np.abs(data.imag[n//2]) <= self._tol):
                        return False
                    # Check for conjugate symmetry
                    if not np.allclose(data[1:n//2], np.conj(data[(n//2)+1:][::-1]), self._tol):
                        return False
                else:
                    # Check if first digit is real (imag ~= 0)
                    if not (np.abs(data.imag[0]) <= self._tol):
                        return False
                    # Check for conjugate symmetry
                    half_1 = data[1:(n+1)//2]
                    half_2 = np.conj(data[((n+1)//2):][::-1])
                    err = np.average(np.abs(half_1-half_2)**2)  # MSE (complex abs)
                    if err > 1e-8:  # some noise tolerance. 1e-8 not chosen empirically!
                        return False
                return True
        else:
            msg = "Invalid dtype"
            raise ValueError(msg)

    @property
    def kind(self):
        return self._kind

    @property
    def reduced(self):
        if self.is_valid:
            if self._reduced:
                return self._data
            n = len(self._data)
            self._even = True if not n % 2 else False
            if self.kind in self.REAL_KINDS:
                data = np.fft.fft(self._data)
            else:
                data = self._data

            if not n % 2:
                r_data = np.r_[data[0], data[1:n//2], data[n//2]]
            else:
                r_data = np.r_[data[0], data[1:(n+1)//2]]
            return r_data
        else:
            err = 1 if self.kind in self.COMPLEX_KINDS else 0
            raise PatternMatchError(err)

    @property
    def n(self):
        _n = len(self.reduced)
        if self._even:
            return 2 + 2*(_n-2)
        else:
            return 1 + 2*(_n-1)


class Container(np.ndarray):
    _SUPPORTED_DATATYPES = (list, np.ndarray)
    _INIT_VARS = {}

    def __new__(cls, data, *args, **kwargs):
        if isinstance(data, cls._SUPPORTED_DATATYPES):
            if not isinstance(data, np.ndarray):
                data = np.array(data, *args, **kwargs)
            else:
                data = data.copy()
        else:
            raise TypeError('Unsupported data type. Must be ndarray-like')
        result = data.view(cls)
        cls.__init_vars(result, cls._INIT_VARS)
        return result


    def __repr__(self):
        s = f'{self.__class__.__name__} {self.view(np.ndarray)}'
        return s

    def __str__(self):
        return self.__repr__()
            

    @classmethod
    def copy_attrs_to_obj(cls, obj_from, obj_to):
        for name in cls._INIT_VARS.keys():
            val = getattr(obj_from, name)
            setattr(obj_to, name, val)
    
    @classmethod
    def zeros(cls, n:tuple, *args, **kwargs) -> Container:
        return cls.__new__(cls, np.zeros(n), *args, **kwargs)
    
    @staticmethod
    def __init_vars(obj: object, vars: dict) -> None:
        # vars should be a dict with entries {[attr_name:str]:[init_val:object]}
        for name, val in vars.items():
            obj.__setattr__(name, val)


class Permutation(Container):
    _VALID_DTYPES = np.sctypes['int'] + np.sctypes['uint']

    def __new__(cls, data, *args, dtype=np.uint64, **kwargs):
        # Validate Pattern
        data = np.array(data, dtype=dtype)
        tracker = np.ones(len(data))
        valid = True
        for i in range(len(data)):
            try:
                tracker[data[i]] = 0
            except KeyError:
                valid = False
                break
        if not valid or np.sum(tracker) != 0:
            raise PatternMatchError(2)

        data = super().__new__(cls, data, *args, **kwargs)
        return data.view(cls)
    
    def invert(self) -> Permutation:
        r = np.empty(len(self))
        r[self] = np.arange(len(self))
        return Permutation(r)
    
    def __invert__(self):
        return self.invert()


class HRR(Container):
    _SUPPORTED_DATATYPES = (list, np.ndarray) 
    _VALID_DTYPES = np.sctypes['complex']
    _SCALARS = (int, float, complex, np.ndarray)
    _DEFAULT_PRINTOPTIONS = {
        'precision':    2,
        'suppress':     True,
        'threshold':    5,
        'edgeitems':    3,
    }
    FIELD = 'R'
    _INIT_VARS = {
        'n': None,
    }

    def __new__(
        cls,
        data, 
        dtype=np.complex128, 
        reduced=False, even=False, 
        printoptions=None, **kwargs):
        # Convert to ndarray
        if isinstance(data, cls._SUPPORTED_DATATYPES):
            if not isinstance(data, np.ndarray):
                data = np.array(data, **kwargs)
            else:
                data = data.copy(**kwargs)
        else:
            raise TypeError('Unsupported data type. Must be ndarray-like')
        pattern = PatternValidator(data, reduced=reduced, even=even)
        if dtype in HRR._VALID_DTYPES:
            r_data = pattern.reduced.astype(dtype)
            hrr = r_data.view(HRR)
            hrr.n = pattern.n
            hrr._printoptions = printoptions
            return hrr
        else:
            msg = "Invalid dtype"
            raise ValueError(msg)
    
    def __add__(self, other):
        if self.n != other.n:
            raise DimensionsNotEqual(self, other)
        hrr = super().__add__(other)
        hrr.n = self.n
        return hrr

    def __iadd__(self, other):
        return self.__add__(other)

    def __radd__(self, other):
        return self.__add__(other)
    
    def __invert__(self):
        i = np.conj(self)
        i.n = self.n
        return i

    def __mul__(self, other):
        if not isinstance(other, HRR._SCALARS):
            if self.n != other.n and other.n is not None:
                raise DimensionsNotEqual(self, other)
        hrr = super().__mul__(other).view(HRR)
        hrr.n = self.n
        return hrr

    def __imul__(self, other):
        return self.__mul__(other)

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __neg__(self):
        i = self * -1
        i.n = self.n
        return i
    
    def __pow__(self, other):
        if not isinstance(other, self._SCALARS):
            err = f'can only exponentiate with scalars ({self._SCALARS})'
            raise TypeError(err)
        hrr = super().__pow__(other)
        hrr.n = self.n
        return hrr
    
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        try:
            print_options = self._printoptions
            if print_options is None:
                print_options = HRR._DEFAULT_PRINTOPTIONS
        except AttributeError:
            print_options = HRR._DEFAULT_PRINTOPTIONS
        with np.printoptions(**print_options):
            s = super().__repr__()
        return s

    def __sub__(self, other):
        return self.__add__(other.__neg__())

    def __isub__(self, other):
        return self.__sub__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __truediv__(self, other):
        if not isinstance(other, self._SCALARS):
            if self.n != other.n:
                raise DimensionsNotEqual(self, other)
        hrr = super().__truediv__(other)
        hrr.n = self.n
        return hrr

    def __itruediv__(self, other):
        return self.__truediv__(other)

    def __rtruediv__(self, other):
        return self.__truediv__(other)
    
    def permute(self, permutation: Permutation | HRR, iterations=1) -> HRR:
        if isinstance(permutation, HRR):
            return self*permutation
        elif isinstance(permutation, Permutation):
            data = self.as_real
            for i in range(abs(iterations)):
                data = data[permutation]
            return HRR(data)
        else:
            err = f'permutation must be of type ({HRR.__name__}, \
                {Permutation.__name__})'
            raise TypeError(err)

    @property
    def as_real(self) -> np.ndarray:
        return np.fft.ifft(self.long).real
    
    @property
    def n(self):
        try:
            return self._n
        except AttributeError:
            self._n = None
        return self._n

    @n.setter
    def n(self, value):
        self._n = value
    
    @property
    def repr(self):
        return self.__repr__()
    
    @property
    def unitary(self) -> HRR:
        hrr = self.copy()
        hrr[hrr != 0] /= np.abs(hrr)[hrr != 0]
        hrr.n = self.n
        return hrr

    @property
    def long(self) -> np.ndarray:
        if not self.n % 2:
            v = np.r_[self, np.conj(self[1:-1][::-1])]
        else:
            v = np.r_[self, np.conj(self[1:][::-1])]
        return v.view(np.ndarray)

    @property
    def is_scalar_array(self) -> bool:
        if self._n is None:
            return True
        return False

    @property
    def sum(self) -> float:
        return self[0].real

    @classmethod
    def identity(cls, n: int, *args, **kwargs) -> HRR:
        return cls.__new__(cls, np.r_[[1], np.zeros(n-1)], *args, **kwargs)

    @classmethod
    def zeros(cls, n: int, *args, **kwargs) -> HRR:
        return cls.__new__(cls, np.zeros(int(n)), *args, **kwargs)

    @staticmethod
    def dot(a:HRR, b:HRR) -> float:
        return (1/a.n)*np.vdot(a.long, b.long).real

    @staticmethod
    def norm(hrr:HRR, ord=2) -> float:
        if ord == 2:
            return np.sqrt(HRR.dot(hrr,hrr))
        elif ord == 1:
            return np.sum(np.abs(hrr.as_real))
        else:
            raise NotImplementedError()

    @staticmethod
    def similarity(a:HRR, b:HRR) -> float:
        n = HRR.dot(a,b)
        d = HRR.norm(a)*HRR.norm(b)
        if d == 0:
            return 0
        return n/d
