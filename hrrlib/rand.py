from hrrlib.core import HRR
from hrrlib.core import Permutation
import numpy as np


def permutation(
        n, 
        seed=None, 
        dtype=np.uint64) -> Permutation:
    np.random.seed(seed)
    p = np.empty(n, dtype=dtype)
    choices = set(range(n))
    for i in range(n):
        c = np.random.choice(list(choices))
        p[i] = c
        choices.remove(c)
    return Permutation(p)


def permutation_element(
        n: int,
        cycle_length: int,
        orthogonal=False,
        seed=None) -> HRR:
    if not orthogonal:
        return __permutation_element(n, cycle_length, seed=seed)
    else:
        return __orthogonal_permutation_element(n, cycle_length, seed=seed)


def plate(n, *args, sqroot=True, seed=None, **kwargs) -> HRR:
    np.random.seed(seed)
    d = np.sqrt(n) if sqroot else 1 / n
    data = np.random.normal(0, 1 / d, n)
    return HRR(data, *args, **kwargs)


def unitary(n: int, seed=None, coset:tuple=(), **kwargs) -> HRR:
    """
    Unitary

    Parameters
    ----------
    n : int
        HRR vector dimension
    seed :
        random seed value
        default None
    third : {'value', 'other'}, optional
        the 3rd param, by default 'value'

    Returns
    -------
    HRR
        a value in a string

    Raises
    ------
    KeyError
        when a key error
    OtherError
        when an other error
    """
    np.random.seed(seed)
    real_angles = [1 + 0j, -1 + 0j]
    n_angles = int(np.ceil((n / 2) - 1))
    n_reals = n - (2 * n_angles)
    angles = np.random.random(n_angles) * 2 * np.pi
    complexs = np.exp(1j * (angles))
    if coset:
        reals = []
        for i in range(n_reals):
            r = coset[i]
            if r:
                r = r/np.abs(r)
            reals.append(r)
    else:
        reals = np.random.choice(real_angles, n_reals)
    return __from_template(reals, complexs, **kwargs)


def __from_template(
        reals,
        complexs,
        dtype=np.complex128,
        reduced=False,
        even=False,
        printoptions=None, **kwargs):
    # return HRR from particularly structured template values
    complex_sym = np.conj(complexs[::-1])
    data = np.r_[reals[0], complexs, reals[1:], complex_sym]
    return HRR(
        data,
        dtype=dtype,
        reduced=reduced,
        even=even,
        printoptions=printoptions,
        **kwargs,
    )


def __get_coprimes(n: int) -> list:
    #
    nums = list(range(int(n)))
    for i in range(2, n):
        if nums[i] == 0:
            continue
        if n / i == int(n / i):
            k = i
            while k < n:
                nums[k] = 0
                k += i
    return [i for i in nums if i != 0]


def __w(n: int, k: int):
    # return root of unity given n, k
    w = (2 * np.pi * (int(k) % int(n))) / n
    if w > float(np.pi):
        w -= (2 * np.pi)
    return w


def __rand_w(cycle_length, subset=None, seed=None):
    # return random root of unity
    np.random.seed(seed)
    if subset is None:
        k = np.random.choice(range(cycle_length))
    else:
        k = np.random.choice(subset)
    np.random.seed(None)
    return __w(cycle_length, k)


def __permutation_element(
        n,
        cycle_length,
        seed=None,
        dtype=np.complex128,
        reduced=False,
        even=False,
        printoptions=None, **kwargs):
    #
    np.random.seed(seed)
    real_angles = np.array([1 + 0j, -1 + 0j])**(((cycle_length % 2) + 1) % 2)
    n_angles = int(np.ceil((n / 2) - 1))
    n_reals = int(n - (2 * n_angles))
    reals = np.random.choice(real_angles, n_reals)

    t = 0
    cp = __get_coprimes(n)
    if cycle_length in cp:
        while not t:
            v = np.array([__rand_w(cycle_length, subset=cp)
                          for _ in range(n_angles)])
            t = np.sum(v)
    else:
        cp = __get_coprimes(cycle_length)
        w = __rand_w(cycle_length, subset=cp)
        wn = w * np.ones(n_angles)
        while not t:
            mask = np.random.choice([0, 1], n_angles)
            t = np.sum(mask)
        v = wn * mask
    complexs = np.exp(1j * v)
    return __from_template(
        reals,
        complexs,
        dtype=dtype,
        reduced=reduced,
        even=even,
        printoptions=printoptions, **kwargs)


def __orthogonal_permutation_element(n: int, cycle_length: int, seed=None):
    np.random.seed(seed)
    cycle_length = abs(cycle_length)
    n = int(abs(n))

    if cycle_length > n:
        err = 'cycle length must be <= n'
        raise ArithmeticError(err)

    generator = int(n / cycle_length)
    if generator != n / cycle_length:
        err = 'cycle length must divide n'
        raise ArithmeticError(err)

    # Build tracker mask
    tracker = np.zeros(n, dtype=np.int64)
    i = 0
    while i < n:
        tracker[i] = 1
        i += generator

    if not n % 2:
        # even
        m = (n - 2) // 2
        r = m % cycle_length
        r_groups = (2 * r + 2) // cycle_length
        flag = False
        if r == m:
            c = [0, 0]
        if not cycle_length % 2:
            if r_groups == 1:
                c = np.random.choice([[0, 1], [1, 0]])
            elif r_groups == 2:
                c = [0, 0]
            elif cycle_length == 2:
                c = np.random.choice([[0, 0], [1, 1]])
            else:
                if m == r:
                    c = [0, 0]
                else:
                    c = [np.random.choice([0, 1]), np.random.choice([0, 1])]
                    flag = True
        elif cycle_length % 2:
            c = [0, 0]
        rtr = tracker.copy()
        if not flag:
            for i in c:
                rtr[i * (n // 2)] = 0
        rem_real_elems = np.where(rtr != 0)[0]
        reals = [(-1 + 0j)**i for i in c]

    else:
        # odd
        m = (n - 1) // 2
        r = m % cycle_length
        rtr = tracker.copy()
        rtr[0] = 0
        c = [0]
        rem_real_elems = np.where(rtr != 0)[0]
        reals = [1 + 0j]

    # Pick values (discard conjugates)
    choices = []
    if c != [0, 0]:
        while len(rem_real_elems):
            c = np.random.choice(range(len(rem_real_elems)))
            a = rem_real_elems[c]
            choices.append(a)
            if a:
                ai = n - a
                ci = np.where(rem_real_elems == ai)[0][0]
            else:
                ci = 0
            rem_real_elems = np.delete(rem_real_elems, [c, ci])
        rem_real_elems = np.array(choices)

    subgroup_elements = np.where(tracker != 0)[0]
    arr = np.empty(m, dtype=np.int64)
    i = 0
    j = len(rem_real_elems)
    while i < m:
        if not (i or j):
            arr[:len(subgroup_elements)] = subgroup_elements
            i += len(subgroup_elements)
            j += len(subgroup_elements)
        elif (not i) and j:
            arr[i:j] = rem_real_elems
            i += j
        else:
            arr[i:j] = subgroup_elements
            i += len(subgroup_elements)
        j += len(subgroup_elements)
    arr = cycle_length * (arr[permutation(m)] / n)
    w = (np.pi * 2) / cycle_length
    complexs = np.exp(1j * w * arr)
    return __from_template(reals, complexs)
