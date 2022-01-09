import numpy as np

from hmeasure.datagen import DataGenBinaryClassifierScores
from hmeasure.h_measure import (
    HMeasure,
    CostRatioDensity
)

EXAMPLE_CLASSIFIER_SCORE_DISTRIBUTION_PARAMS = {
    'class0_alpha': 1.5,
    'class0_beta': 2.0,
    'class1_alpha': 2.0,
    'class1_beta': 2.0
}


def example_h_measure():
    c0_sample_size = 100
    c1_sample_size = 80 # slightly imbalanced classification problem
    dg_bcs = DataGenBinaryClassifierScores(class_params=EXAMPLE_CLASSIFIER_SCORE_DISTRIBUTION_PARAMS,
                                           c0_sample_size=c0_sample_size,
                                           c1_sample_size=c1_sample_size)
    score_samples = dg_bcs.generate_samples()
    crd = CostRatioDensity()
    hm = HMeasure(cost_distribution=crd)
    hm.plot(score_samples)
    hm.plot_scores(score_samples)


def assert_monotonic(vals: np.array, tol=1e-5):
    vdiff = np.diff(vals)
    vneg = vdiff[vdiff<tol]
    if len(vneg) != 0:
        raise AssertionError("vals are not monotonic")


def example_monotonicity():
    c0_sample_size = 1000
    c1_sample_size = 800  # slightly imbalanced classification problem
    params = EXAMPLE_CLASSIFIER_SCORE_DISTRIBUTION_PARAMS
    c0_betas = np.linspace(2.0, 20.0, num=5)
    h_vals = np.zeros(shape=len(c0_betas))
    for i, c0_beta in enumerate(c0_betas):
        params['class0_beta'] = c0_beta
        dg_bcs = DataGenBinaryClassifierScores(class_params=params,
                                               c0_sample_size=c0_sample_size,
                                               c1_sample_size=c1_sample_size)
        score_samples = dg_bcs.generate_samples()
        crd = CostRatioDensity()
        hm = HMeasure(cost_distribution=crd)
        _, _, h_val = hm.h_measure(score_samples)
        h_vals[i] = h_val

    assert_monotonic(h_vals)


def main():
    example_h_measure()
    example_monotonicity()


if __name__ == "__main__":
    main()