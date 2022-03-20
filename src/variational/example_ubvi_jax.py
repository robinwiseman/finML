import jax
import jax.numpy as jnp
from jax import (
    grad,
    value_and_grad,
    jit,
    vmap,
    random
)
from jax.nn import elu
from jax.scipy.stats.norm import pdf as norm_pdf
from jax.lax import stop_gradient, dynamic_slice

from optax import adam
from flax.core import FrozenDict

import numpy as np
from functools import partial
from scipy.optimize import nnls

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


PRNGKEY = random.PRNGKey(41)


@jit
def p_cauchy(X, mu, gamma):
    return (1/(jnp.pi*gamma))*(gamma*gamma/(jnp.power(X-mu,2) + gamma*gamma))


@jit
def p_cauchy_mixture(X):
    mu1 = 0.5
    mu2 = -0.4
    gamma1 = 0.3
    gamma2 = 0.2
    mix1 = 0.4
    mix2 = 1.0-mix1
    f1 = p_cauchy(X,mu1,gamma1)
    f2 = p_cauchy(X,mu2,gamma2)
    return mix1*f1 + mix2*f2


@partial(jit, static_argnums=(1))
def normal_sample(key, n_samples, mu, cov):
    samples = random.normal(key, shape=(n_samples,))
    sig = jnp.sqrt(cov)
    samples = samples*sig + mu
    probs = normal_prob(samples, mu, sig)
    return samples, probs

@jit
def normal_prob(y, mu, sig):
    return norm_pdf(y, loc=mu, scale=sig) # N(mu, sig) *NOT* N(mu, sig^2)


class BoostingLayer:
    def __init__(self, layer_params):
        key, subkey = random.split(PRNGKEY)
        self.mu = random.normal(subkey, shape=(1,1))
        _, subkey = random.split(key)
        self.cov_raw = random.normal(subkey, shape=(1,1))
        if layer_params is not None:
            # start close to previous, but not exactly at
            old_mu, old_cov = layer_params
            self.mu = old_mu + 0.1*self.mu
            self.cov_raw = old_cov + 0.1*self.cov_raw

    def __call__(self):
        return self.mu, self.cov_raw


@partial(jit, static_argnums=(1))
def h_sample(key, sample_size, mu, cov):
    # NOTE: these are samples for h^2 ~ N(h_mu, h_cov) (not h):
    return normal_sample(key, sample_size, mu, cov)

@jit
def gaussian_convolution(mu_p, sig2_p, mu_q, sig2_q):
    # return <p|q> where
    # p^2 ~ N(mu_p, sig2_p),
    # q^2 ~ N(mu_q, sig2_q), and
    # <p|q> = \int\limits_{-\infty}^{\infty} p(x)q(x)dx
    vexp = jnp.power((mu_p - mu_q), 2) / (2*sig2_p + 2*sig2_q)
    res = jnp.exp(-0.5*vexp)*jnp.power(16*sig2_p*sig2_q, 0.25)/jnp.power(2*(sig2_p+sig2_q), 0.5)

    return res


@jit
def apply_gaussian_conv(g_params, h_mu, h_cov):
    g_mu, g_cov = transform_raw_params(g_params)
    # only let h_mu and h_cov participate in learning, keeping g_mu and g_cov fixed
    return gaussian_convolution(h_mu, h_cov, stop_gradient(g_mu), stop_gradient(g_cov))

@jit
def transform_raw_params(params):
    mu, cov = params
    cov = elu(cov)+(1.0 + 1e-5) # ensure positive covariance
    return mu, cov


@partial(jit, static_argnums=(1))
def g_objective(prng_key, h_sample_size, params, all_parms):
    previous_params = all_parms['previous_params']
    boosting_step = all_parms['boosting_step']
    h_mu, h_cov = transform_raw_params(params)
    # NOTE: these are samples for h^2 ~ N(h_mu, h_cov) (not h):
    h_samples, h_probs = h_sample(prng_key, h_sample_size, h_mu, h_cov)
    h_probs = jnp.sqrt(h_probs)
    # E_{h^2}[f(x)/h(x)] Monte Carlo estimate:
    f_probs = jnp.multiply(jnp.sqrt(p_cauchy_mixture(h_samples)), 1/h_probs)
    f_h = jnp.sum(f_probs)/h_sample_size

    gauss_conv = partial(apply_gaussian_conv, h_mu=h_mu, h_cov=h_cov)
    gc = vmap(gauss_conv)
    filler = vmap(lambda _: np.array([[0.0]]))
    g_h = jax.lax.cond(boosting_step>0,
                       lambda x: gc(x),
                       lambda x: filler(x),
                       jnp.array(previous_params))

    g_h = jnp.transpose(g_h)[0,0]

    return g_h, f_h


def argmin_h_loss(prng_key, h_sample_size, params, all_parms):
    g_h, f_h = g_objective(prng_key, stop_gradient(h_sample_size), params, all_parms)
    boosting_step = all_parms['boosting_step']
    lambdas = all_parms['lambdas']
    d_fg = all_parms['d_fg']
    objective = f_h
    h_gbar = jnp.sum(jnp.multiply(g_h,
                                  stop_gradient(dynamic_slice(lambdas, (0,), (boosting_step,)))))
    f_gbar = jnp.sum(jnp.multiply(stop_gradient(dynamic_slice(d_fg, (0,), (boosting_step,))),
                                  stop_gradient(dynamic_slice(lambdas, (0,), (boosting_step,)))))
    objective += f_gbar*h_gbar*-1

    # minimise the negative log objective
    objective /= jnp.power(1 - h_gbar*h_gbar, 0.5)
    objective = -jnp.sign(objective)*jnp.log(jnp.sign(objective)*objective)

    return objective


def update(prng_key, h_sample_size, params, all_parms, lr):
    loss_val, grad_val = value_and_grad(argmin_h_loss, argnums=(2))(prng_key, h_sample_size, params, all_parms)
    params = tuple([param-lr*grad_v for param, grad_v in zip(params, grad_val)])
    return params, loss_val


class UniversalBoostingVariationalModel:
    def __init__(self, learning_rate: float = 1e-3, boosting_steps: int = 20, argmin_steps: int = 50,
                 optimiser = adam, h_sample_size: int = 50, fg_sample_size: int = 500):
        self.learning_rate = learning_rate
        self.boosting_steps = boosting_steps
        self.argmin_steps = argmin_steps
        self.h_sample_size = h_sample_size
        self.fg_sample_size = fg_sample_size
        self.nnls_iter_factor = 30
        self.optimiser = optimiser
        # these are intentionally not traceable - just np, rather than jnp
        # they remain fixed once learned, and aren't variable in gradient descent
        self.params = [(0.0, 0.0)]
        self.lambdas = np.zeros(shape=self.boosting_steps)
        self.Z = np.zeros(shape=(self.boosting_steps, self.boosting_steps))
        self.d_fg = np.zeros(shape=self.boosting_steps)

    def all_params(self, boosting_step):
        # gather parameters for passing around to enable passing them as static arguments, but hashable
        return FrozenDict({
            'previous_params': self.params,
            'boosting_step': boosting_step,
            'd_fg': self.d_fg,
            'lambdas': self.lambdas
        })

    def updates_with_argmin_h(self, boosting_step, layer_params, key):
        # now that the latest g_{boosting_step} has been fitted, update:
        # (1) the Z-matrix of <g,g> components
        # (2) the d-vector of <f,g> components
        # (3) the lambda-vector
        self.update_Z(boosting_step, layer_params)
        self.update_d(boosting_step, layer_params, key)
        self.update_lambda(boosting_step)

    def update_Z(self, boosting_step, layer_params):
        mu_gbs, cov_gbs = transform_raw_params(layer_params)
        for i in range(boosting_step+1):
            mu_gi, cov_gi = transform_raw_params(self.params[i])
            gg = gaussian_convolution(mu_gbs, cov_gbs, mu_gi, cov_gi)
            self.Z[i,boosting_step] = gg
            self.Z[boosting_step, i] = gg

    def update_d(self, boosting_step, layer_params, key):
        g_mu, g_cov = transform_raw_params(layer_params)
        # NOTE: these are samples for g^2 ~ N(g_mu, g_cov) (not g)
        g_samples, g_probs = h_sample(key, self.fg_sample_size, g_mu, g_cov)
        g_probs = jnp.sqrt(g_probs)
        # E_{g^2}[f(x)/g(x)] Monte Carlo estimate
        f_probs = jnp.multiply(jnp.sqrt(p_cauchy_mixture(g_samples)), 1/g_probs)
        f_g = jnp.sum(f_probs) / self.fg_sample_size
        self.d_fg[boosting_step] = f_g

    def update_lambda(self, boosting_step):
        current_z = self.Z[:boosting_step+1,:boosting_step+1]
        current_d = self.d_fg[:boosting_step+1]
        Linv = jnp.linalg.inv(jnp.linalg.cholesky(current_z))
        Linvd = jnp.matmul(Linv, np.transpose(current_d))
        beta, _ = nnls(Linv, -np.transpose(Linvd), maxiter=self.nnls_iter_factor * Linv.shape[1])
        beta = np.reshape(beta, newshape=(1, beta.shape[0]))
        bd = beta + current_d
        lams = np.transpose(jnp.matmul(jnp.linalg.inv(current_z), np.transpose(bd)))
        lams = lams/np.sqrt(np.dot(bd, np.transpose(lams))[0,0])
        self.lambdas = lams[0]

    def q_prob(self, X, boosting_step):
        g = 0
        for i in range(boosting_step+1):
            mu, cov = transform_raw_params(self.params[i])
            g_i_p = normal_prob(X, mu, np.sqrt(cov))
            g += np.sqrt(g_i_p) * self.lambdas[i]

        return g*g

    def validation(self, boosting_step):
        lams = self.lambdas
        norm_check = 0
        for i in range(boosting_step + 1):
            for j in range(boosting_step + 1):
                mu_i, cov_i = transform_raw_params(self.params[i])
                mu_j, cov_j = transform_raw_params(self.params[j])
                norm_check += lams[i]*lams[j]*gaussian_convolution(mu_i, cov_i, mu_j, cov_j)

        print(f'q norm check: {norm_check}')
        return norm_check

    def fit(self, verbose=True):
            '''
            Treat each new distribution function to be learned in the forward stage additive modelling of boosting
            as a separate learnable function on which gradient descent is performed. Post learning of that function
            its learnable parameters then remain fixed on further boosting steps.
            '''
            optim = self.optimiser(learning_rate=self.learning_rate)
            lr_start = self.learning_rate
            layer_params = None
            key = PRNGKEY
            boosting_step = 0
            loss_diff = 1e-9 # loss change threshold to early terminate at
            q_tol = 1e-6 # acceptable max normalization error for q
            while boosting_step < self.boosting_steps:
                try:
                    # initialize first set of params, else start close to one of the previous
                    if layer_params is not None:
                        if boosting_step % 5 == 0:
                            random_layer = 0 # anchor on the first fit periodically to avoid wandering too far
                        else:
                            random_layer = np.random.randint(0,high=boosting_step)
                        layer_params = self.params[random_layer]

                    layer = BoostingLayer(layer_params)
                    layer_params = layer()
                    lr = lr_start
                    # update the pseudo-random number generator key at each boosting step, but keep it constant during
                    # gradient descent (gradient-descend on a constant noisy surface)
                    key, subkey = random.split(key)
                    am_step = 0
                    loss_prior = 0
                    while am_step < self.argmin_steps+1:
                        # gradient descent to find the optimal mu, cov for the latest h distribution to add to the boosting
                        # collection of distributions in forward stagewise additive modelling
                        all_parms = self.all_params(boosting_step)
                        layer_params, loss = update(subkey, self.h_sample_size, layer_params, all_parms, lr)
                        lp = transform_raw_params(layer_params)

                        if verbose:
                            if (am_step % 1000) == 0:
                                print(f"boosting_step: {boosting_step}: argmin_step: {am_step}: loss: {loss}: lr: {lr}: ", end='')
                                print(f"mu: {float(lp[0])}: cov: {float(lp[1])}")

                        # gradients = tape.gradient(loss, self.trainable_variables)
                        # self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                        lr = lr*0.99999
                        optim = self.optimiser(learning_rate=lr)
                        am_step += 1
                        if np.abs(loss-loss_prior) < loss_diff:
                            am_step = self.argmin_steps+1 # early terminate
                        loss_prior = loss

                    # Check if this boosting step managed to arrive at a minimum within the max number of descent
                    # steps. If it didn't repeate the boosting step, starting from a different point, rather than
                    # adding parameters that aren't at a minimum
                    if np.abs(loss - loss_prior) < loss_diff:
                        # updates using the optimised h (i.e. the new additional g)
                        if boosting_step == 0:
                            self.params[boosting_step] = layer_params
                        else:
                            self.params.append(layer_params)
                        self.updates_with_argmin_h(boosting_step, layer_params, subkey)
                        q_validation = self.validation(boosting_step)

                        if verbose:
                            print(f"completed boosting step: {boosting_step}")
                            print(f"lambdas: {self.lambdas[:boosting_step + 1]}")
                            print(f"Z: {self.Z[:boosting_step + 1, :boosting_step + 1]}")
                            print(f"d: {self.d_fg[:boosting_step+1]}")
                            self.plotfit(boosting_step)
                        if np.abs(q_validation-1.0) < q_tol:
                            boosting_step += 1
                        else:
                            self.params.pop()  # remove this last set up params
                            self.params.pop()
                            boosting_step -= 1
                            # increase sample sizes to try to improve accuracy of q-normalization
                            # + increase number of nnls iterations
                            self.h_sample_size = int(self.h_sample_size*1.1)
                            self.fg_sample_size = int(self.fg_sample_size*1.1)
                            self.nnls_iter_factor = int(self.nnls_iter_factor*1.5)
                            print(f"Rolling back two boosting steps, q normalisation not within tolerance")
                    else:
                        print(f"Repeating boosting step, no minimum found within max steps: {boosting_step}")
                except Exception as ex:
                    print(ex)
                    print(f'Repeating boosting step:{boosting_step}')

    def plotfit(self, boosting_step):
        font = {'family': 'normal','size': 18}
        matplotlib.rc('font', **font)
        # for 1-d distribution visualisation
        X = np.random.uniform(low=-3, high=3, size=500)
        p = p_cauchy_mixture(X)
        q = self.q_prob(X, boosting_step)
        df_vals = {'x': X, 'p': p, 'q': q[0]}
        df = pd.DataFrame(df_vals)
        df.loc[:, 'pq_diff'] = df.loc[:, 'p'] - df.loc[:, 'q']
        ax1 = df.plot(kind='scatter', x='x', y='p', color='r', figsize=(16, 12), label='p_exact')
        ax1.legend()
        ax2 = df.plot(kind='scatter', x='x', y='q', color='g', ax=ax1, figsize=(16, 12), label='q_approx')
        ax2.legend()
        ax3 = df.plot(kind='scatter', x='x', y='pq_diff', color='b', ax=ax1, figsize=(16, 12), label='p-q diff')
        ax3.legend()
        plt.show()


def main():
    lr = 5e-3
    boosting_steps = 50
    argmin_steps = 100000
    h_sample_size = 20000
    fg_sample_size = 100000
    ubvi = UniversalBoostingVariationalModel(learning_rate = lr,
                                             boosting_steps = boosting_steps,
                                             argmin_steps = argmin_steps,
                                             optimiser = adam,
                                             h_sample_size = h_sample_size,
                                             fg_sample_size = fg_sample_size
                                            )
    ubvi.fit()


if __name__ == "__main__":
    main()