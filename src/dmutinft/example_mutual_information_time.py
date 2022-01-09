import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from dmutinft.datagen import DataGenMVNTransformed
from dmutinft.dmit import DifferentialMutualInformationTime


def yz_cov(dim, yz_correl, zz_correl):
    z_cov = np.eye(dim - 1) * (1 - zz_correl) + np.ones((dim-1, dim-1)) * zz_correl
    y_cov_elem = np.zeros((dim,dim))
    y_cov_elem[:, 0] = yz_correl
    y_cov_elem[0, :] = yz_correl
    y_cov_elem[0, 0] = 1
    cov = y_cov_elem
    cov[1:, 1:] = z_cov
    return cov


def plot(df, y_nonlinear):
    y_postfixes = {
        True : 'y non-linear',
        False : 'y linear'
    }
    y_postfix = y_postfixes[y_nonlinear]
    font_size = 16
    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=font_size)
    plt.rc('axes', labelsize=font_size)
    plt.rc('figure', titlesize=font_size)
    ax = df.plot.box(figsize=(12, 8))
    ax.set_xlabel("yz correlation")
    ax.set_ylabel("D({y}|{z})")
    ax.set_title("D({y}|{z}) vs :yz correlation : " + f"{y_postfix}")


def example_mut_int_time():
    x_series_dim = 4 # dim(y) + dim(z)
    sample_size = 1000
    mean = np.zeros(x_series_dim)
    yz_correls = np.linspace(0.0, 0.8, num=10)
    zz_correl = 0.6
    num_tests = 10

    for y_nonlinear in [False, True]:
        all_res = []
        for test in range(num_tests):
            res = []
            for yz_correl in yz_correls:
                cov = yz_cov(x_series_dim, yz_correl=yz_correl, zz_correl=zz_correl)
                dg = DataGenMVNTransformed(mean, cov=cov, sample_size=sample_size)
                x = dg.generate_samples(y_nonlinear=y_nonlinear) # simulate y and z
                z = x[:,1:]
                y = x[:,0]
                y = np.reshape(y, newshape=(y.shape[0], 1))
                dmir = DifferentialMutualInformationTime()
                mi_rate = dmir.time(y,z)
                res.append({'yz_corr': yz_correl, 'mi_rate': mi_rate})

            df = pd.DataFrame(res)
            df['yz_corr'] = df['yz_corr'].round(3)
            df = df.set_index('yz_corr')
            df = df.T.reset_index().drop(columns=['index'])
            all_res.append(df)
            print(".", end="")

        df = pd.concat(all_res)
        plot(df, y_nonlinear)

    return df


def main():
    example_mut_int_time()


if __name__=='__main__':
    main()