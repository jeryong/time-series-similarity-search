import numpy as np
import pandas as pd
import time


def SPRING(X, Y, eta):
    n, m = len(X), len(Y)
    ts, te = 0, 0
    ssq_store, mat, a_min_store = [], [], []
    d, _d, s, _s = ([0] * m for i in range(4))
    _d[0] = (Y[0] - X[0]) ** 2
    d_min = 10000
    for i in range(1, m):
        _d[i] = _d[i - 1] + (Y[i] - X[0]) ** 2
    for t in range(1, n):
        d[0] = (Y[0] - X[t]) ** 2
        s[0] = t
        for i in range(1, m):
            f0, f1, f2 = d[i - 1], _d[i - 1], _d[i]
            if f0 <= f1 and f0 <= f2:
                min_f = f0
                s[i] = s[i - 1]
            elif f1 <= f0 and f1 <= f2:
                min_f = f1
                s[i] = _s[i - 1]
            else:
                min_f = f2
                s[i] = _s[i]
            d[i] = min_f + (Y[i] - X[t]) ** 2
        mat.append(_d)
        if d_min <= eta:
            if all([d[i] >= d_min or s[i] > te for i in range(m)]) == True:
                mat_s = ts - t
                mat_e = (te - t + 1, None)[(te - t + 1) >= 0]
                mat_temp = mat[mat_s:mat_e]
                if mat_e != None:
                    mat = mat[-1:mat_e]
                mat_temp = np.array(mat_temp)
                _len = mat_temp.shape[0]
                l, a_min = 0, [0] * _len
                for r in range(_len):
                    l += np.argmin(mat_temp[r, l:None])
                    a_min[r] = l
                # a_min_value = mat_temp[np.arange(len(mat_temp)), a_min]
                a_min_store.append(a_min)
                ssq_store.append((ts, te, d_min))
                d_min = 10000
                for i in range(m):
                    if s[i] <= te:
                        d[i] = 10000
        if d[m - 1] <= eta and d[m - 1] < d_min:
            ts, te, d_min = s[m-1], t, d[m-1]
        _d, _s = d[:], s[:]
    return ssq_store, a_min_store


def pre_processing(df, col, ref_s, ref_e):
    x = df.iloc[:, col].tolist()
    y = x[df.index.get_loc(ref_s): df.index.get_loc(ref_e) + 1]
    return x, y


def get_trait(ref_t, ssq_store, a_min_store):
    ssq_t_store = []

    for ssq in ssq_store:
        if ref_t >= ssq[0] and ref_t <= ssq[1]:
            rel_ref_t = ref_t - ssq[0]
            for i, a_min in enumerate(a_min_store):
                ref_s = ssq_store[i][0]
                lbound = np.searchsorted(a_min, rel_ref_t)
                ubound = np.searchsorted(a_min, rel_ref_t, 'right')
                if ubound - lbound == 0:
                    lbound_start = np.searchsorted(a_min, a_min[lbound-1])
                    n = 1
                    while True:
                        lbound_start_check = np.searchsorted(
                            a_min, a_min[lbound - 1] - n)
                        if lbound_start_check == lbound_start:
                            break
                        lbound_start = lbound_start_check
                        n += 1
                    rel_ssq_t = lbound_start + (lbound - lbound_start) * \
                        (rel_ref_t - a_min[lbound_start]
                         ) // (a_min[lbound] - a_min[lbound_start])
                else:
                    rel_ssq_t = lbound
                ssq_t_store.append(ref_s + rel_ssq_t)
            return ssq_t_store
    return 'Trait out of bounds.'


def refine_trait(ref_t, ssq_t_store, Y, testrange, testsize):
    tr = testrange // 2
    ts = testsize // 2
    y_sample = Y[ref_t - ts: ref_t + ts + 1]

    def con_diff(list):
        con_diff_arr = []
        for prev, curr in zip(list, list[1:]):
            con_diff_arr.append(curr - prev)
        return con_diff_arr
    y_sample_cd = con_diff(y_sample)
    for i, ssq_t in enumerate(ssq_t_store):
        cd_rmse = []
        for j in range(ssq_t - tr, ssq_t + tr + 1):
            y_test = Y[j - ts: j + ts + 1]
            y_test_cd = con_diff(y_test)
            cd_rmse.append(
                sum([(ys - yt) ** 2 for ys, yt in zip(y_sample_cd, y_test_cd)]))
        refined_ssq_t = cd_rmse.index(min(cd_rmse)) - tr + ssq_t
        ssq_t_store[i] = refined_ssq_t
    return ssq_t_store