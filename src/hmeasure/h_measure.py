import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta
from scipy.integrate import quad
from typing import Dict, Optional
from inspect import ismethod


class CostRatioDensity:
    def __init__(self, c_density_obj=None):
        if c_density_obj is None:
            self.c_density_obj = beta(2,2) # default H-Measure choice
        else:
            if not (self.has_method(c_density_obj, 'pdf') and
                    self.has_method(c_density_obj, 'cdf')):
                raise NotImplementedError(f"""density object: {c_density_obj} is missing pdf or cdf""")
            self.c_density_obj = c_density_obj

    def __call__(self, cost):
        '''
        :param cost: is the cost ratio of the costs for incorrect classification of each class
        cost := (1+(c1/c0))^(-1), and so lies in [0,1]. Suitable distribution functions should be
        defined on the same domain.
        :return: the probability density of the cost
        '''
        return self.c_density_obj.pdf(cost)

    def uc(self, cost):
        return cost*self.c_density_obj.pdf(cost)

    def u1mc(self, cost):
        return (1-cost)*self.c_density_obj.pdf(cost)

    def cdf(self, cost):
        return self.c_density_obj.cdf(cost)

    @staticmethod
    def has_method(obj, method):
        return hasattr(obj, method) and ismethod(getattr(obj, method))


class HMeasure:
    def __init__(self, cost_distribution: CostRatioDensity, class0_prior: float = None,
                 class1_prior: float = None):
        self.cost_distribution = cost_distribution
        self.class0_prior = class0_prior
        self.class1_prior = class1_prior
        self.c_merged = None
        self.c_classes = None
        self.roc_curve = None
        self.convex_hull = None
        self.int_components = None
        self.H = None

    def h_measure(self, scores: Dict[str, np.array]):
        '''
        :param scores: a dict of class_0 and class_1 scores, each as numpy arrays
        :return:
        '''
        c0_scores = scores.get('class_0', None)
        c1_scores = scores.get('class_1', None)
        self.c0_num = len(c0_scores)
        self.c1_num = len(c1_scores)
        if (self.class0_prior is None) or (self.class1_prior is None):
            # Optionally pass in population priors (to handle case where model training is over/under sampled
            # for one class relative to the other.
            # If no priors are passed, then calculate priors from the score populations.
            self.class0_prior = self.c0_num/(self.c0_num+self.c1_num)
            self.class1_prior = self.c1_num/(self.c0_num+self.c1_num)
        c0_scores = np.sort(c0_scores)
        c1_scores = np.sort(c1_scores)
        if self.H is None:
            self.c_merged, self.c_classes = self._merge_scores(c0_scores, c1_scores)
            self.roc_curve = self._build_roc(self.c_merged, self.c_classes)
            self.convex_hull, self.int_components = self._build_chull(self.roc_curve)
            self.H = self._build_H(self.int_components)

        return self.roc_curve, self.convex_hull, self.H

    def plot(self, scores: Optional[Dict[str, np.array]] = None):
        if self.convex_hull is None:
            # if h_measure was not already built, run it now, but to do so requires the scores to be passed in
            if scores is None:
                raise AttributeError(f"""Scores must be passed in to build the H-Measure""")
            roc_curve, convex_hull, h_val = self.h_measure(scores)
        else:
            roc_curve = self.roc_curve
            convex_hull = self.convex_hull
            h_val = self.H

        figsize = (8,6)
        fontsize = 16
        rc_random = np.zeros(roc_curve.shape)
        rc_random[:, 0] = roc_curve[:, 0]
        rc_random[:, 1] = roc_curve[:, 0] # the roc curve for a random classifier
        df_rc = pd.DataFrame({'F1': roc_curve[:, 0],
                              'F0': roc_curve[:, 1]})
        df_ch = pd.DataFrame({'ch1': convex_hull[:, 0],
                              'convex_hull': convex_hull[:, 1]})
        df_rdm = pd.DataFrame({'rand1': rc_random[:, 0],
                               'random': rc_random[:, 1]})
        ax1 = df_rc.plot(x='F1', y='F0', figsize=figsize, color='b', kind='scatter', legend=True, fontsize=16)
        ax2 = df_ch.plot(x='ch1', y='convex_hull', figsize=figsize, color='r', kind='scatter', ax=ax1, legend=True,
                         fontsize=16)
        ax3 = df_ch.plot(x='ch1', y='convex_hull', figsize=figsize, color='r', kind='line', ax=ax1, legend=True,
                         fontsize=16)
        ax4 = df_rdm.plot(x='rand1', y='random', figsize=figsize, color='g', kind='line', ax=ax1, legend=True,
                          fontsize=16)
        ax1.legend(loc=4, fontsize=20)
        ax1.set_ylabel('F0', fontdict={'fontsize': fontsize})
        ax1.set_xlabel('F1', fontdict={'fontsize': fontsize})
        plt.title(f"""H-measure: {np.round(h_val, 4)}""", fontdict={'fontsize': fontsize})
        plt.show()

    @staticmethod
    def plot_scores(scores: Optional[Dict[str, np.array]] = None, set_scale=True):
        dfs0 = pd.DataFrame({'score': scores['class_0']})
        dfs0['class'] = 'c0'
        dfs1 = pd.DataFrame({'score': scores['class_1']})
        dfs1['class'] = 'c1'
        df = pd.concat([dfs0, dfs1])
        if set_scale:
            sns.set_theme(font_scale=1.3)
        g = sns.FacetGrid(df, col="class", height=3.5, aspect=1.1)
        for i in range(2):
            g.axes[0, i].set_xlabel('score', fontsize=16)
            g.axes[0, i].set_ylabel('count', fontsize=16)
            g.axes[0, i].title._text = f"""class = {i}"""
            g.axes[0, i]._left_title.text = f"""class = {i}"""
            g.axes[0, i]._right_title.text = f"""class = {i}"""
        g.map(sns.histplot, "score")
        plt.show()

    def _merge_scores(self, c0: np.array, c1: np.array):
        c_merged_scores = np.zeros(self.c0_num+self.c1_num)
        c_merged_classes = np.zeros(self.c0_num+self.c1_num, dtype=int)
        c0_i = 0
        c1_i = 0
        cm_i = 0
        while (c0_i < self.c0_num) or (c1_i < self.c1_num):
            if c0_i < self.c0_num and c1_i < self.c1_num:
                if c0[c0_i] <= c1[c1_i]:
                    c_merged_scores[cm_i] = c0[c0_i]
                    c_merged_classes[cm_i] = 0
                    c0_i += 1
                    cm_i += 1
                elif c1[c1_i] <= c0[c0_i]:
                    c_merged_scores[cm_i] = c1[c1_i]
                    c_merged_classes[cm_i] = 1
                    c1_i += 1
                    cm_i += 1
            elif c0_i < self.c0_num and c1_i >= self.c1_num: # loop to fill in the remaining c0
                c_merged_scores[cm_i] = c0[c0_i]
                c_merged_classes[cm_i] = 0
                c0_i += 1
                cm_i += 1
            elif c1_i < self.c1_num and c0_i >= self.c0_num: # loop to fill in the remaining c1
                c_merged_scores[cm_i] = c1[c1_i]
                c_merged_classes[cm_i] = 1
                c1_i += 1
                cm_i += 1

        return c_merged_scores, c_merged_classes

    def _build_roc(self, c_scores, c_classes):
        rc_points = np.array([[0.0,0.0]])
        i = 0
        rc_point = np.array([0.0, 0.0])
        while i < len(c_scores):
            i_score = c_scores[i]
            duplicate_i = np.argwhere(c_scores == i_score) # handle case of potentially multiple scores of the same value
            duplicate_i_classes = c_classes[duplicate_i]
            duplicate_0 = len(np.argwhere(duplicate_i_classes == 0))
            duplicate_1 = len(np.argwhere(duplicate_i_classes == 1))
            rc_point += [duplicate_1/self.c1_num, duplicate_0/self.c0_num] # class1 on the 'x-axis', class0 on the 'y-axis' : [x,y]
            rc_points = np.append(rc_points, [rc_point], axis=0)
            i += len(duplicate_i)

        return rc_points

    def _build_chull(self, roc_curve):
        i = 0
        chull = np.array([[0.0,0.0]]) # the coordinates of the convex hull in roc curve space
        cval_loc_prior = np.array([0.0,0.0])
        cval_prior = 0.0
        int_components = None
        while i < roc_curve.shape[0]-1:
            cvals = self._cvals(roc_curve, i)
            c_argmin = np.argmin(cvals)
            cval_loc = roc_curve[i+c_argmin+1] # the location of the c value on the roc curve
            chull = np.append(chull, [cval_loc], axis=0)
            # Note: for points on the roc_curve, e.g. cval_loc, cval_loc[0] corresponds to class 1, cval_loc[1] to
            # class 0. Collect elements required for subsequent integration to generate the H-measure
            if int_components is None:
                # collect [c_{i+1},c_i,r1_{i+1},r0_{i+1}] on the convex hull
                int_components = np.array([[cvals[c_argmin], cval_prior, cval_loc_prior[0], cval_loc_prior[1]]])
            else:
                int_components = np.append(int_components, np.array([[cvals[c_argmin], cval_prior, cval_loc_prior[0], cval_loc_prior[1]]]), axis=0)
            cval_loc_prior = cval_loc
            cval_prior = cvals[c_argmin]
            i += c_argmin+1

        int_components = np.append(int_components, np.array([[1.0, cval_prior, cval_loc_prior[0], cval_loc_prior[1]]]), axis=0)
        # drop rows where cval_{i+1} == cval_i as they don't contribute to the integral
        int_components = int_components[int_components[:, 0] != int_components[:, 1]]

        return chull, int_components

    def _cvals(self, roc_curve, current_idx):
        current_point = roc_curve[current_idx]
        return np.array([self._cval(current_point, i_point) for i_point in roc_curve[current_idx+1:,:]])

    def _cval(self, current_point, i_point):
        # Note: for points, point[0] corresponds to class 1, point[1] to class 0
        return self.class1_prior*(i_point[0]-current_point[0])/(self.class0_prior*(i_point[1]-current_point[1]) + self.class1_prior*(i_point[0]-current_point[0]))

    def _build_L(self, int_components):
        L = 0
        for i, int_vals in enumerate(int_components):
            L += self._int_L_step(int_vals)

        return L

    def _int_L_step(self, int_vals):
        r0i = int_vals[3]
        r1i = int_vals[2]
        ci = int_vals[1]
        cip1 = int_vals[0]
        int_uc = quad(self.cost_distribution.uc, ci, cip1)[0]
        int_u1mc = quad(self.cost_distribution.u1mc, ci, cip1)[0]

        L_step = self.class0_prior*(1-r0i)*int_uc + self.class1_prior*r1i*int_u1mc

        return L_step

    def _L_max(self):
        L_max = quad(self.cost_distribution.uc, 0, self.class1_prior)[0]*self.class0_prior
        L_max += quad(self.cost_distribution.u1mc, self.class1_prior, 1)[0]*self.class1_prior
        return L_max

    def _build_H(self, int_vals):
        L = self._build_L(int_vals)
        L_max = self._L_max()
        return 1 - L/L_max

