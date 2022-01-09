import numpy as np
from typing import Optional, Any


class DifferentialEntropyRate:
    def __init__(self, entropy_rate_method: str = 'max_ent'):
        ent_methods = {
            'max_ent': self._max_ent
        }
        self._entropy_rate_method = entropy_rate_method
        self._entropy_rate_func = ent_methods.get(self._entropy_rate_method)

    def entropy_rate(self, z, method_parms: Optional[Any] = None):
        if method_parms is None:
            method_parms = {}
        return self._entropy_rate_func(z, method_parms)

    def _max_ent(self, z, method_parms):
        z_dim = z.shape[1]
        N = z.shape[0]  # number of time steps in each series
        z = self.normalise(z)
        max_lag = method_parms.get('cov_max_lag', self._max_cov_lag(N))
        block_cov_p = self._covariance(z, max_lag)
        block_cov_p_1 = self._covariance(z,max_lag-1)
        ent_rate = (z_dim/2)*np.log2(2*np.pi*np.exp(1)) + 0.5*np.log2(np.linalg.det(block_cov_p)/np.linalg.det(block_cov_p_1))
        if method_parms.get('clear_cache', True):
            self.covs = {} # clear cache post returning rate by default (optionally retain it)

        return ent_rate

    @staticmethod
    def normalise(z):
        # normalising as below ensures that h({z}) <= N, with equality iff:
        # (i) z is memoryless (independent samples),
        # (ii) the coodinate processes of z are independent
        # (iii) z is a Gaussian process
        # h({z}) < N => temporal dependency and/or cross-dependency and/or non-Gaussianity
        N = z.shape[0]
        z_mean = np.sum(z, axis=0, keepdims=True) / N
        z -= z_mean
        z_std = np.std(z, axis=0, keepdims=True)
        z /= z_std
        z *= np.sqrt(2/(np.pi*np.exp(1)))
        return z

    def _covariance(self, z, max_lag):
        self.covs = {} # cache of previously calculated covs at given lag
        z_dim = z.shape[1] # number of series
        self.cov_block = np.zeros(shape=(z_dim*(max_lag+1), z_dim*(max_lag+1))) # full covariance block matrix at all lags

        for i in range(max_lag+1):
            for j in range(i, max_lag+1):
                cov = self._covariance_elem(z,abs(i-j))
                self.cov_block[i*z_dim:(i+1)*z_dim,j*z_dim:(j+1)*z_dim] = np.transpose(cov) # i-j <= 0
                self.cov_block[j*z_dim:(j+1)*z_dim,i*z_dim:(i+1)*z_dim] = cov # i-j > 0

        return self.cov_block

    def _covariance_elem(self, z, lag):
        # returns one block of the block covariance matrix : corresponding to the given lag
        cov = self.covs.get(lag, None)
        if cov is None:
            N = z.shape[0]
            for i in range(lag,N):
                z1 = z[lag:N]
                z2 = z[:N-lag]
                self.covs[lag] = self._autocovariance(z1,z2)
                return self.covs[lag]
        else:
            return cov

    @staticmethod
    def _autocovariance(z1,z2):
        res = np.zeros(shape=(z1.shape[1], z1.shape[1]))
        for i in range(z1.shape[0]):
            res += np.outer(z1[i], z2[i])

        res = res / z1.shape[0]

        return res

    @staticmethod
    def _max_cov_lag(T):
        return int(np.floor(12*np.power(T/100,0.25)))


class DifferentialMutualInformationTime:
    def __init__(self, entropy_rate_method: str = 'max_ent'):
        self.dentr = DifferentialEntropyRate(entropy_rate_method=entropy_rate_method)

    def time(self, y, z, check=False):
        yz = np.concatenate((y, z), axis=1)
        h_y = self.dentr.entropy_rate(y)
        h_z = self.dentr.entropy_rate(z)
        h_yz = self.dentr.entropy_rate(yz)

        if check:
            assert h_y <= y.shape[1]
            assert h_z <= z.shape[1]
            assert h_yz <= yz.shape[1]

        return 1/(h_y + h_z - h_yz)
