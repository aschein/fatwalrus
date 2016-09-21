import numpy as np
import numpy.random as rn

from genmodel import GenModel


class PMF(GenModel):
    """Poisson matrix factorization with Gamma priors."""
    def __init__(self, n_rows=30, n_cols=20, n_components=5, shape=0.5, rate=1.0):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_components = n_components
        self.shape = shape
        self.rate = rate
        GenModel.__init__(self)

    def generate_state(self):
        state = {}
        state['Theta_DK'] = rn.gamma(self.shape, 1. / self.rate, size=(self.n_rows, self.n_components))
        state['Phi_KV'] = rn.gamma(self.shape, 1. / self.rate, size=(self.n_components, self.n_cols))
        tmp = rn.poisson(np.einsum('dk,kv->dvk', state['Theta_DK'], state['Phi_KV']))
        state['Y_DVK'] = np.ma.MaskedArray(tmp, mask=None)
        return state

    def reconstruct(self, state={}):
        Theta_DK = state['Theta_DK'] if 'Theta_DK' in state.keys() else self.Theta_DK
        Phi_KV = state['Phi_KV'] if 'Phi_KV' in state.keys() else self.Phi_KV
        return np.dot(Theta_DK, Phi_KV)

    def generate_data(self, state={}, mask=np.ma.nomask):
        Y_DVK = state['Y_DVK'] if 'Y_DVK' in state.keys() else self.Y_DVK
        assert isinstance(Y_DVK, np.ma.core.MaskedArray)
        return np.ma.MaskedArray(Y_DVK.data.sum(axis=2), mask=mask)

    def _init_state_vars(self):
        GenModel._init_state_vars(self)
        if hasattr(self, 'data'):
            self.Y_DVK.mask = np.repeat(self.data.mask[:, :, np.newaxis], self.n_components, axis=2)
            self._update_Y_DVK()

    def _init_data(self, data):
        GenModel._init_data(self, data)
        if hasattr(self, 'Y_DVK'):
            self.Y_DVK.mask = np.repeat(data.mask[:, :, np.newaxis], self.n_components, axis=2)
            self._update_Y_DVK()

    def _init_cache(self):
        self.Y_DK = np.zeros((self.n_rows, self.n_components), dtype=int)
        self.Y_KV = np.zeros((self.n_components, self.n_cols), dtype=int)

    def _update_Y_DVK(self):
        for d, v in zip(*self.data.nonzero()):
            prob_K = self.Theta_DK[d] * self.Phi_KV[:, v]
            prob_K /= prob_K.sum()
            self.Y_DVK[d, v, :] = rn.multinomial(self.data[d, v], prob_K)
        # self._cache_Y_DVK(),

    def _cache_Y_DVK(self):
        self.Y_DK[:] = self.Y_DVK.sum(axis=1)
        self.Y_KV[:] = self.Y_DVK.sum(axis=0).T

    def _update_Theta_DK(self):
        post_shape_DK = self.shape + self.Y_DK
        post_rate_DK = self.rate + np.dot(~self.data.mask, self.Phi_KV.T)
        self.Theta_DK[:] = rn.gamma(post_shape_DK, 1. / post_rate_DK)

    def _update_Phi_KV(self):
        post_shape_KV = self.shape + self.Y_KV
        post_rate_KV = self.rate + np.dot(self.Theta_DK.T, ~self.data.mask)
        self.Phi_KV[:] = rn.gamma(post_shape_KV, 1. / post_rate_KV)


if __name__ == '__main__':
    pmf = PMF(n_rows=8, n_cols=4, n_components=2, rate=0.25)
    pmf.geweke_test(mask_type=None, verbose=False, schedule={'Theta_DK': 0, 'Phi_KV': 0})
