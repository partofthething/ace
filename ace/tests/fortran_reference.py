"""
Line-by-line Python transliteration of Friedman's FORTRAN smooth, supsmu, scale, and mace.

This intentionally mirrors the FORTRAN control flow rather than being idiomatic so
it can serve as an independent reference for the main implementation. Only unit
weights, non-periodic variables (iper=1), and orderable variables (l=1) are supported.
"""
# pylint: skip-file
import numpy as np

SPANS = (0.05, 0.2, 0.5)
BIG, SML, EPS = 1e20, 1e-7, 1e-3


def smooth(x, y, span, iper, vsmlsq):
    n = len(x)
    smo = np.zeros(n)
    acvr = np.zeros(n)
    xm = ym = var = cvar = fbw = 0.0
    ibw = int(0.5 * span * n + 0.5)
    if ibw < 2:
        ibw = 2
    it = 2 * ibw + 1
    for j in range(it):
        xti = x[j]
        fbo = fbw
        fbw += 1.0
        xm = (fbo * xm + xti) / fbw
        ym = (fbo * ym + y[j]) / fbw
        tmp = fbw * (xti - xm) / fbo if fbo > 0 else 0.0
        var += tmp * (xti - xm)
        cvar += tmp * (y[j] - ym)
    for j in range(1, n + 1):  # 1-based
        out = j - ibw - 1
        inn = j + ibw
        if not (out < 1 or inn > n):
            xto, xti = x[out - 1], x[inn - 1]
            fbo = fbw
            fbw -= 1.0
            tmp = fbo * (xto - xm) / fbw if fbw > 0 else 0.0
            var -= tmp * (xto - xm)
            cvar -= tmp * (y[out - 1] - ym)
            xm = (fbo * xm - xto) / fbw
            ym = (fbo * ym - y[out - 1]) / fbw
            fbo = fbw
            fbw += 1.0
            xm = (fbo * xm + xti) / fbw
            ym = (fbo * ym + y[inn - 1]) / fbw
            tmp = fbw * (xti - xm) / fbo if fbo > 0 else 0.0
            var += tmp * (xti - xm)
            cvar += tmp * (y[inn - 1] - ym)
        a = cvar / var if var > vsmlsq else 0.0
        smo[j - 1] = a * (x[j - 1] - xm) + ym
        if iper <= 0:
            continue
        h = 1.0 / fbw if fbw > 0 else 0.0
        if var > vsmlsq:
            h += (x[j - 1] - xm) ** 2 / var
        acvr[j - 1] = 0.0
        a = 1.0 - h
        if a > 0:
            acvr[j - 1] = abs(y[j - 1] - smo[j - 1]) / a
        elif j > 1:
            acvr[j - 1] = acvr[j - 2]
    # tie averaging
    j = 0
    while j < n:
        j0 = j
        sy = smo[j]
        cnt = 1.0
        while j < n - 1 and x[j + 1] <= x[j]:
            j += 1
            sy += smo[j]
            cnt += 1
        if j > j0:
            smo[j0:j + 1] = sy / cnt
        j += 1
    return smo, acvr


def variance_threshold(x):
    """Compute vsmlsq like supsmu."""
    n = len(x)
    i = n // 4
    j = 3 * i
    scale = x[j - 1] - x[i - 1]
    while scale <= 0:
        if j < n:
            j += 1
        if i > 1:
            i -= 1
        scale = x[j - 1] - x[i - 1]
    return (EPS * scale) ** 2


def supsmu(x, y, span=0.0, alpha=0.0):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if not x[-1] > x[0]:
        return np.full(n, y.mean())
    vsmlsq = variance_threshold(x)
    if span > 0:
        return smooth(x, y, span, 1, vsmlsq)[0]
    prim, ress = [], []
    for s in SPANS:
        sm, cv = smooth(x, y, s, 1, vsmlsq)
        prim.append(sm)
        ress.append(smooth(x, cv, SPANS[1], -1, vsmlsq)[0])
    best = np.zeros(n)
    for j in range(n):
        resmin = BIG
        for i in range(3):
            if ress[i][j] < resmin:
                resmin = ress[i][j]
                best[j] = SPANS[i]
        if 0 < alpha <= 10 and resmin < ress[2][j] and resmin > 0:
            best[j] += (SPANS[2] - best[j]) * max(SML, resmin / ress[2][j]) ** (10 - alpha)
    sbest = smooth(x, best, SPANS[1], -1, vsmlsq)[0]
    out = np.zeros(n)
    for j in range(n):
        s = min(max(sbest[j], SPANS[0]), SPANS[2])
        f = s - SPANS[1]
        if f < 0:
            f = -f / (SPANS[1] - SPANS[0])
            out[j] = (1 - f) * prim[1][j] + f * prim[0][j]
        else:
            f = f / (SPANS[2] - SPANS[1])
            out[j] = (1 - f) * prim[1][j] + f * prim[2][j]
    return smooth(x, out, SPANS[0], -1, vsmlsq)[0]


def scale(ty, tx, eps, maxit):
    """Conjugate-gradient linear regression of ty on tx columns (mace 'scale')."""
    n, p = tx.shape
    sw = float(n)
    sc1 = np.zeros(p)
    nit = 0
    while True:
        nit += 1
        sc5 = sc1.copy()
        sc4 = np.zeros(p)
        h = 0.0
        for it in range(1, p + 1):
            r = ty - tx @ sc1
            sc2 = -2.0 * (r @ tx) / sw
            s = sc2 @ sc2
            if s <= 0:
                break
            if it == 1:
                sc3 = -sc2
                h = s
            else:
                gama = s / h
                h = s
                sc3 = -sc2 + gama * sc4
            u = tx @ sc3
            delta = (u @ r) / (u @ u)
            sc1 = sc1 + delta * sc3
            sc4 = sc3
        if np.max(np.abs(sc1 - sc5)) < eps or nit >= maxit:
            break
    return tx * sc1


def mace(xs, y, delrsq=0.01, maxit=20, nterm=3):
    y = np.asarray(y, float)
    X = np.column_stack([np.asarray(xi, float) for xi in xs])
    n, p = X.shape
    ty = y - y.mean()
    ty /= np.sqrt(np.mean(ty ** 2))
    tx = X - X.mean(axis=0)
    my = np.argsort(y, kind='stable')
    mx = [np.argsort(X[:, i], kind='stable') for i in range(p)]
    tx = scale(ty, tx, delrsq, p)
    rsq = 0.0
    it = 0
    ct = [100.0] * nterm
    nt = 0
    while True:
        it += 1
        nit = 0
        while True:
            rsqi = rsq
            nit += 1
            z5 = ty - tx.sum(axis=1)
            for i in range(p):
                k = mx[i]
                z1 = z5[k] + tx[k, i]
                z3 = supsmu(X[k, i], z1)
                z3 -= z3.mean()
                sv = 1.0 - np.mean((z1 - z3) ** 2)
                if sv <= rsq:
                    continue
                rsq = sv
                tx[k, i] = z3
                z5[k] = z1 - z3
            if p == 1 or rsq - rsqi <= delrsq or nit >= maxit:
                break
        z1 = tx[my].sum(axis=1)
        z3 = supsmu(y[my], z1)
        z3 -= z3.mean()
        z3 /= np.sqrt(np.mean(z3 ** 2))
        ty = np.empty(n)
        ty[my] = z3
        rsq = 1.0 - np.mean((ty - tx.sum(axis=1)) ** 2)
        nt = nt % nterm
        ct[nt] = rsq
        nt += 1
        if max(ct) - min(ct) <= delrsq or it >= maxit:
            break
    return tx, ty, rsq
