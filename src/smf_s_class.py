# -*- coding:utf-8 -*-
'''
@Author: Licong Liu
@email: liulicong@mail.bnu.edu.cn
@Date: 2022/04/30
@Description:
    SMFS - A new shape model fitting based phenology detection method.
    Please cite: Detecting crop phenology from vegetation index time-series data by improved
    shape model fitting in each phenological stage.
'''


import numpy as np
from numba import jit


@jit(nopython=True)
def shape_change_faster(t, p0, tshift, xscale, aug_ts, trans_ts):
    '''
    Numba accelerated version of the shape_model_change function.
    :return:
    '''
    t = t * xscale + (1 - xscale) * p0 + tshift * xscale
    for i, t_ in enumerate(t):
        if t_ <= 0:
            t[i] = 0
        if t_ >= 365:
            t[i] = 365
        trans_ts[i] = aug_ts[int(t[i] * 100)]
    return


@jit(nopython=True)
def pcc(x, y, length):
    '''
    Numba accelerated version of the pearson correlation coefficient calculation program.
    :param x: timeseries 1
    :param y: timeseries 1
    :param length: the length of timeseries
    :return:
    '''
    xmm, ymm = x - np.mean(x), y - np.mean(y)
    tp1, tp2, tp3 = 0.0, 0.0, 0.0
    for i in range(length):
        tp1 += xmm[i] * ymm[i]
        tp2 += xmm[i] * xmm[i]
        tp3 += ymm[i] * ymm[i]
    tp2, tp3 = np.sqrt(tp2), np.sqrt(tp3)
    return tp1 / (tp2 + 1e-6) / (tp3 + 1e-6)


class SMFS:
    '''
    This is a implement of SMFS algorithms in Python.
    The demonstration of phenology detection can be find in reference.ipynb.
    The demonstration of width determination can be find in reference_width.ipynb.
    '''
    P0 = 0  # Reference phenological period, which needs to be specified by the user.
    REF_VT = None  # Reference vegetation index curve, required for specifying.
    T = None  # The time of the point of the reference vegetation index curve. It needs to be equally spaced.

    WIN = 45  # default matching half windows, centered on reference phenology. It can be changed by user.

    DTOL = 1e-1  # day tolerance
    MAXITER = 10  # maximum iteration numbers
    RTOL = 0.8  # After match, if the correlation coefficient is less than RTOL, the result is discarded.
    BND_XSCALE = (0.7, 1.3, )  # The initial xscale parameter search range
    BND_TSHIFT = (-30, 30, )  # The initial tshift parameter search range
    TSHIFT_ITER_SEARCH_RANGE = 4  # Parameter search range of tshift parameter after the first loop
    XSCALE_ITER_SEARCH_RANGE = 0.2  # Parameter search range of xscale parameter after the first loop
    TSHIFT_SEARCH_STEP = 1  # tshift parameter search step size
    XSCALE_SEARCH_STEP = 0.05  # xscale parameter search step size
    DINT = 8  # The interval of time series, the algorithm can only process time series of equal interval temporarily.

    REF_VT_AUG = None  # A array to hold the augmented array.
    T_AUG = np.arange(0, 365, 0.01)  # How the Augmented array was  sampled.
    TEMP_TRANS = np.zeros((46, ))  # A temporary array that holds changed shape model.

    def __init__(self, ref_vt, p0, t):
        self.REF_VT = ref_vt
        self.T = t
        self.DINT = int(self.T[1] - self.T[0])
        self.REF_VT_AUG = np.interp(self.T_AUG, t, ref_vt)
        self.P0 = p0
        self.length = len(ref_vt)

    def shape_change(self, tshift, xscale):
        '''
        In shape model change procedure, we need acquired following function:
        g^(x) = g(xscale * x + (1 - xscale) * pi0 + xscale * tshift)

        In computers, we store curves in discrete. So the curve we actually get is:
        g^(1), g^(2), g^(3) =
            g(xscale * 1 + (1 - xscale) * pi0 + xscale * tshift),
            g(xscale * 2 + (1 - xscale) * pi0 + xscale * tshift),
            g(xscale * 3 + (1 - xscale) * pi0 + xscale * tshift)

        For shape model change functions, xscale, tshift, and p0 are all specified. Taking xscale, tshift, p0
        as 1, 0.5, 0 as an example, what we want to obtain is actually the following curve:
        g^(1), g^(2), g^(3) =
            g(1.5),
            g(2.5),
            g(3.5)

        This goal can be easily achieved by interpolation. For example, the current <shape_change>
        function is a common interpolation based implement of shape model change.
        However, this method is not efficient, we used an another method to conduct shape model change.
        Please refer to <shape_change_faster>
        '''
        t = self.T * xscale + (1 - xscale) * self.P0 + tshift * xscale
        return np.interp(t, self.T, self.REF_VT)

    def shape_change_faster(self, tshift, xscale):
        '''
         We use an new array (we call augmented array) to speed up the operation. The augmented array is generated
         each time when the instance of the class is created, and only needs to be calculated once for a reference curve.

         Specifically, the augmented array achieves the following purposes:
         g(1.5) = self.REF_VT_AUG[1500], g(2.5) = self.REF_VT_AUG[2500], g(3.5) = self.REF_VT_AUG[3500]

         Therefore, when changing the shape model, we replaced interpolation operation with the array manipulate operation,
         which greatly speeds up the operation speed. The new method can also be accelerated by numba.

        :param tshift: tshift is the time shift factor. Please refer to eq.7 in paper.
        :param xscale: xscale is the scaling factors in the time dimensions. Please refer to eq.7 in paper.
        :return:
        '''
        # The function called here is accelerated by numba
        shape_change_faster(self.T, self.P0, tshift, xscale, self.REF_VT_AUG, self.TEMP_TRANS)
        return self.TEMP_TRANS

    def loss(self, tshift, xscale, tar_ts):
        # After the shape model is changed, the result was saved in self.TEMP_TRANS.
        shape_change_faster(self.T, self.P0, tshift, xscale, self.REF_VT_AUG, self.TEMP_TRANS)

        # Following by the Eq.8 in paper, we only need points in (P0-tshift-w, P0-tshift+w)
        t0, length = int((self.P0 - tshift - self.WIN) / self.DINT), int(self.WIN / self.DINT) * 2 + 1

        # Avoiding index out of bounds
        if t0 < 0:
            t0 = 0
        if length >= self.length - t0 - 1:
            length = self.length - t0 - 1

        # sampling the points in array
        x, y = self.TEMP_TRANS[t0: t0 + length], tar_ts[t0: t0 + length]

        # calculating the pearson correlation coefficient
        r = pcc(x, y, length)
        return 1 - r

    def tshift_search(self, tshift, xscale, tar_ts):
        '''

        The explanation for manual search used in the method:

        Experienced users may wonder why we use manual search to find the parameters optimization. We have tried various automatic multi-parameter optimization algorithms in Python. However, we found that the effect is not as good as our manual-search method.

        The first reason is that these methods tend to fall into local optima. The problem is prevalent in the original approach of SMF. In the original SMF method, the xscale parameter is easier to change the RMSE between the reference and target than the tshift parameter. Thus the SMF method is more inclined to optimize the xscale parameter optimization process. And then fall into the local optimum of xscale and fail to optimize tshift. In the SMFS method, we reduce the correlation between the xscale parameter and the tshift parameter, which alleviates this problem. However, in practice, when using the automatic optimization algorithm, the situation of falling into a local optimum still occasionally occurs.

        Another counter-intuitive and more important reason is that the speed of these optimization algorithms is even much slower than that of our manual-searched algorithms. Because our phenology detection does not need to pursue high precision in parameter search, these optimization algorithms always spend much time deciding whether this phenological period is 98.45 days or 98.46 days. Of course, some algorithms provide the method to jump out of the optimization by controlling the termination condition, for example, the minimum threshold for the change of Loss gradient. However, whether it is the LOSS of the correlation coefficient or the LOSS of the RMSE, they are indirect indicators for phenology. It is difficult for us to measure to what extent the correlation coefficient is optimized to ensure that the error of phenology is within the acceptable level. Maybe 0.99 will guarantee a fair phenology accuracy. Maybe 0.999 will do. The number depends on the shape of the specific optimization target curve and the curve smoothness. Therefore, it is difficult for us to balance the automated method's efficiency and accuracy.

        Therefore, in the SMFS method, we finally use the manual-search method for optimization. However, we also know that manual search is not a final solution for this problem. When manual-search, to balance the efficiency and accuracy of the algorithm, we use a few parameters to control the parameter optimization process. We believe that there must be more suitable methods to help us complete the parameter optimization. We will continue to research and update the code constantly.

        We are looking forward to giving better and more beautiful solutions in the future.

        :param tshift: tshift is the time shift factor. Please refer to eq.7 in paper.
        :param xscale: xscale is the scaling factors in the time dimensions. Please refer to eq.7 in paper.
        :param tar_ts: Target vegetation index time series for phenology monitor.
        :return:
        '''

        # If it is the first time to search for parameters, search for parameters in the entire bnd range
        if tshift > 1e4:
            search_locs = np.arange(self.BND_TSHIFT[0], self.BND_TSHIFT[1])
            xscale = 1

        # If it is not the first time to search for parameters, it will only search the parameters
        # within the self.TSHIFT_ITER_SEARCH_RANGE range of the current position.
        # Using max, min operation to avoid index out of bounds.
        else:
            search_locs = np.arange(np.max((tshift - self.TSHIFT_ITER_SEARCH_RANGE, self.BND_TSHIFT[0], )),
                                    np.min((tshift + self.TSHIFT_ITER_SEARCH_RANGE + 1, self.BND_TSHIFT[1], )))
        r_arr = np.zeros(len(search_locs))
        for i, v in enumerate(search_locs):
            r_arr[i] = self.loss(v, xscale, tar_ts)
        return search_locs[np.argmin(r_arr)]

    def xscale_search(self, tshift, xscale, tar_ts):
        '''
        :param tshift: tshift is the time shift factor. Please refer to eq.7 in paper.
        :param xscale: xscale is the scaling factors in the time dimensions. Please refer to eq.7 in paper.
        :param tar_ts: Target vegetation index time series for phenology monitor.
        :return:
        '''
        # If it is the first time to search for parameters, search for parameters in the entire BND range
        if xscale > 1e4:
            search_locs = np.arange(self.BND_XSCALE[0], self.BND_XSCALE[1], self.XSCALE_SEARCH_STEP)
        # If it is not the first time to search for parameters, it will only search the parameters
        # within the self.XSCALE_ITER_SEARCH_RANGE range of the current position.
        # Using max, min operation to avoid index out of bounds.
        else:
            search_locs = np.arange(np.max((xscale - self.XSCALE_ITER_SEARCH_RANGE, self.BND_XSCALE[0], )),
                                    np.min((xscale + self.XSCALE_ITER_SEARCH_RANGE + 1, self.BND_XSCALE[1], )),
                                    self.XSCALE_SEARCH_STEP)
        loss_arr = np.zeros(len(search_locs))
        for i, v in enumerate(search_locs):
            loss_arr[i] = self.loss(tshift, v, tar_ts)
        i_min = np.argmin(loss_arr)
        return search_locs[i_min], 1 - loss_arr[i_min]

    def doit(self, tar_ts):
        # Used to save the result of tshift in every loop
        tshift_previous = np.inf
        # Initialize tshift, xscale with np.inf, so that the algorithm search the entire BND range on the first loop
        tshift, xscale = np.inf, np.inf
        # Record the number of loops and the correlation coefficient of last loop
        iter, r = 0, - np.inf
        while iter < self.MAXITER:  # Used self.MAXITER to avoid falling into an infinite loop
            # Correct the relative time shift of ref_ts and tar_ts
            tshift = self.tshift_search(tshift, xscale, tar_ts)
            # If the different of tshift of two loops is within the self.DTOL, stop the loop
            if abs(tshift_previous - tshift) < self.DTOL:
                break
            tshift_previous = tshift
            # Scaling ref_ts to make it closer to target_curve
            xscale, r = self.xscale_search(tshift, xscale, tar_ts)
            iter += 1
        if r < self.RTOL:
            return 0
        else:
            return self.P0 - tshift
