from sympy import ring
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polyclasses import ANP
from sympy.polys.polyerrors import DomainError
from sympy.utilities.pytest import raises

from sympy.polys.splittingfield import split_field

def test_split_field():
    R, x = ring('x', ZZ)

    f = x**2 + 1

    raises(DomainError, lambda: split_field(f))

    R, x = ring('x', QQ)

    f = x**2 + 1
    minpoly = x**2 + 1

    assert split_field(f) == (minpoly, [
        ANP([1, 0], [1, 0, 1], QQ),
        ANP([-1, 0], [1, 0, 1], QQ)])

    f = x**3 - 11
    minpoly = x**6 + 3267

    assert split_field(f) == (minpoly, [
        ANP([QQ(-1, 198), 0, 0, QQ(1, 2), 0], [1, 0, 0, 0, 0, 0, 3267], QQ),
        ANP([QQ(1, 99), 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 3267], QQ),
        ANP([QQ(-1, 198), 0, 0, QQ(-1, 2), 0], [1, 0, 0, 0, 0, 0, 3267], QQ)])
