from sympy.core.compatibility import xrange

from sympy import Dummy
from sympy.polys.factorization_alg_field import _alpha_to_z
from sympy.polys.polyerrors import DomainError
from sympy.polys.rootoftools import RootOf


def _subst(p, beta, ring):
    r"""
    Return `p(\beta, x)` for a polynomial `p(x, \gamma)` over
    `\mathbbQ(\gamma)`.

    """
    z = Dummy('z')
    zring_QQ = ring.clone(symbols=ring.symbols + (z,), domain=ring.domain.domain)
    q = _alpha_to_z(p, zring_QQ)

    zring = ring.clone(symbols=ring.symbols + (z,))
    q = q.set_ring(zring).evaluate(zring.gens[0], beta)

    newp = ring.zero
    for monom, coeff in q.iterterms():
        newp[monom] = coeff

    return newp


def _subst_alg_element(a, alpha, domain):
    r"""
    Substitute the algebraic element `\gamma` in `a \in \mathbb Q(\gamma)`
    by `\alpha`.

    """
    newa = domain.zero
    a = a.rep
    n = len(a)

    for i in xrange(n):
        newa += a[i]*alpha**(n - (i + 1))

    return newa


def split_field(f):
    """
    Compute the splitting field of an irreducible univariate polynomial over
    `\mathbb Q`.

    References
    ==========

    1. [Trager76]_

    """
    ring = f.ring
    domain = ring.domain
    if not domain.is_QQ:
        raise DomainError("ground domain must be a rational field")

    roots = []
    polys = [f]

    minpoly = f
    newminpoly = f
    gamma = RootOf(minpoly.as_expr(), 0)
    index = 0

    newdomain = domain.algebraic_field(gamma)
    newring = ring.clone(domain=newdomain)
    gamma = newdomain.convert(gamma)
    beta = gamma
    x = newring.gens[0]

    while True:
        polys[index] = polys[index].set_ring(newring) // (x - beta)
        roots.append(beta)

        newfactors = []
        k = 0
        Bpoly = None

        for p in polys:
            s, g, norm = p.sqf_norm()
            _, normfactors = norm.factor_list()

            for q, _ in normfactors:
                gfactor = g.gcd(q.set_ring(newring))

                if q.degree() > newminpoly.degree():
                    newminpoly = q
                    index = k
                    news = s
                    Bpoly = gfactor

                g //= gfactor
                factor = gfactor.compose(x, x + s*gamma)

                if factor.degree() == 1:
                    roots.append(-factor.coeff(1) / factor.LC)
                else:
                    newfactors.append(factor)
                    k += 1

        if not newfactors:
            return newminpoly, roots

        newgamma = RootOf(newminpoly.as_expr(), 0)
        newdomain = domain.algebraic_field(newgamma)
        newring = ring.clone(domain=newdomain)

        x = newring.gens[0]
        newgamma = newdomain.convert(newgamma)

        Bpoly = _subst(Bpoly, newgamma, newring)
        l = minpoly.set_ring(newring).gcd(Bpoly)
        alpha = -l.coeff(1) / l.LC

        beta = newgamma - news*alpha

        roots = [_subst_alg_element(root, alpha, newdomain) for root in roots]

        for i, factor in enumerate(newfactors):
            newfactor = newring.zero
            for monom, coeff in factor.iterterms():
                newfactor[monom] = _subst_alg_element(coeff, alpha, newdomain)
            newfactors[i] = newfactor

        polys = newfactors
        minpoly = newminpoly
        gamma = newgamma
