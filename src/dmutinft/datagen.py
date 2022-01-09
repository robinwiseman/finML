import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from fracdiff import Fracdiff


class DataGenMVNTransformed:
    def __init__(self, mean: np.array, cov: np.array, sample_size: int = 100, fdiff_order: float = 0.8, fdiff_window=10):
        self.mean = mean
        self.cov = cov
        self.sample_size = sample_size
        self.fdiff_order = fdiff_order # suitably high to ensure stationarity
        self.fdiff_window = fdiff_window
        self.series = None

    def generate_samples(self, y_nonlinear=True):
        z = np.random.multivariate_normal(self.mean, self.cov, size=self.sample_size)
        if y_nonlinear:
            z[:, 0] = z[:, 0] * z[:, 0] * z[:,0]
        z = np.cumsum(z, axis=0)
        fd = Fracdiff(self.fdiff_order, window=self.fdiff_window, mode='valid', window_policy='fixed')
        self.series = fd.fit_transform(z)
        return self.series

    def plot(self, series):
        series_dim = self.mean.shape[0]
        df = pd.DataFrame({f's{i}' : series[:,i] for i in range(series_dim)})
        df.plot(figsize=(12,8))
        plt.show()
