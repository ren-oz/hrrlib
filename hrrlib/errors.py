class DimensionsNotEqual(Exception):
    def __init__(self, a, b, msg=None):
        if msg is None:
            msg = f'operands must have same dimension ({a.n}), ({b.n})'
        else:
            msg = msg + f'({a}), ({b})'
        super().__init__(msg)


class PatternMatchError(Exception):
    def __init__(self, mismatch_code, *args, **kwargs):
        codes_msg = [
            'Reals mismatch',
            'Complex mismatch',
            'Invalid permutation'
        ]
        super().__init__(codes_msg[mismatch_code], *args, **kwargs)


class InverseError(ArithmeticError):
    def __init__(self):
        err = "divide by zero (true inverse element does not exist)"
        super().__init__(err)
