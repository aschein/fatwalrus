import inspect
import sys
import time
import warnings

import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from collections import defaultdict
from contextlib import contextmanager
from sklearn.base import BaseEstimator


######################################################################################
#########                           Geweke Testing                           #########
######################################################################################

def geweke(model, n_samples=10000, mask_type=None, n_itns=10, schedule={}, verbose=False, **kwargs):
    forward_dict = defaultdict(lambda: defaultdict(list))
    backward_dict = defaultdict(lambda: defaultdict(list))
    try:
        for i in xrange(n_samples):
            if i % 100 == 0:
                print 'ITERATION %d' % i

            state = model.generate_state()
            data = model.generate_data(state=state, mask=None)
            data.mask = generate_mask(data, mask_type=mask_type)

            calc_funcs(data.compressed(), name='data', func_dict=forward_dict)
            for var in model._STATE_VARS:
                if schedule[var] is not None and schedule[var] <= n_itns:
                    calc_funcs(state[var], name=var, func_dict=forward_dict)

            model._total_itns = 0
            model.set_state(state)
            model.fit(data, n_itns=n_itns, schedule=schedule, verbose=verbose, **kwargs)
            state = model.get_state()
            data = model.generate_data(state, mask=data.mask)

            calc_funcs(data.compressed(), name='data', func_dict=backward_dict)
            for var in model._STATE_VARS:
                if schedule[var] is not None and schedule[var] <= n_itns:
                    calc_funcs(state[var], name=var, func_dict=backward_dict)

    except KeyboardInterrupt:
        if raw_input('\nMake plots? y/n') == 'y':
            for var in sorted(forward_dict.keys()):
                pp_plot(forward_dict[var], backward_dict[var], title=var)

        if raw_input('Continue test? y/n') == 'y':
            geweke(model=model,
                   n_samples=n_samples - i,
                   n_itns=n_itns,
                   schedule=schedule,
                   verbose=verbose,
                   **kwargs)
        else:
            sys.exit()

    for var in sorted(forward_dict.keys()):
        pp_plot(forward_dict[var], backward_dict[var], title=var)


def pp_plot(F_dict, G_dict, title=None, xlabel=None, ylabel=None, file_name=None):
    """Generates a P-P plot for given functions F and G.

    Arguments:
        F -- List of samples from function F.
        G -- List of samples from function G.
    """
    n_plots = len(F_dict.keys())
    x = int(np.ceil(np.sqrt(n_plots)))
    fig = plt.figure()
    if title is not None:
        fig.suptitle(title, size=20)
    gs = gridspec.GridSpec(x, x, wspace=0.0, hspace=0.0)
    if xlabel is None:
        xlabel = 'CDF of generative samples'
    if ylabel is None:
        ylabel = 'CDF of inferential samples'

    for n, (i, j) in enumerate(np.ndindex(x, x)):
        if n < n_plots:
            key = F_dict.keys()[n]
            F = np.array(F_dict[key])
            G = np.array(G_dict[key])
            F.sort()

            F_cdf = [np.mean(F < f) for f in F]
            G_cdf = [np.mean(G < f) for f in F]
            ax = fig.add_subplot(gs[i, j])
            ax.set_title(key)
            ax.plot(F_cdf, G_cdf, 'b.', lw=0.005)
            ax.plot([-0.05, 1.05], [-0.05, 1.05], 'g--', lw=1.5)

        if i == x - 1:
            ax.set_xlabel(xlabel)
            plt.setp(ax.get_xticklabels(), fontsize=8)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)

        if j == 0:
            ax.set_ylabel(ylabel)
            plt.setp(ax.get_yticklabels(), fontsize=8)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)

    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)
        plt.close()


def calc_funcs(X, name, func_dict):
    if isinstance(X, np.ma.core.MaskedArray):
        X = X.compressed()
    if np.isscalar(X) or X.size == 1:
        func_dict[name][name].append(np.mean(X))
    else:
        func_dict[name]['E[X]'].append(np.mean(X))
        func_dict[name]['G[X]'].append(np.exp(np.mean(np.log(X + 1e-300))))
        func_dict[name]['V[X]'].append(np.var(X))
        func_dict[name]['max[X]'].append(np.max(X))


def generate_mask(data, mask_type=0.3):
    if mask_type is None:
        mask = None
    elif np.isscalar(mask_type):
        assert mask_type > 0 and mask_type < 1
        mask = (rn.random(size=data.shape) < mask_type).astype(bool)
    else:
        raise NotImplementedError
    return mask

######################################################################################
#########                          GenModel Class                            #########
######################################################################################


class GenModel(BaseEstimator):
    def __init__(self):
        self._STATE_VARS = get_state_vars(self)
        self._total_itns = 0

    def generate_state(self, **kwargs):
        raise NotImplementedError

    def generate_data(self, state={}, mask=None):
        raise NotImplementedError

    def _init_cache(self):
        pass

    def _init_state_vars(self):
        self._init_cache()
        self.set_state(self.generate_state())

    def _init_data(self, data):
        if isinstance(data, np.ndarray):
            data = np.ma.array(data, mask=None)
        assert isinstance(data, np.ma.core.MaskedArray)
        self.data = data.copy()

    def get_state(self):
        return dict([(var, np.ma.copy(getattr(self, var))) for var in self._STATE_VARS])

    def set_state(self, state):
        self._init_cache()
        for var in state.keys():
            if var in self._STATE_VARS:
                setattr(self, var, state[var])
                if hasattr(self, '_cache_%s' % var):
                    getattr(self, '_cache_%s' % var)()
            else:
                warnings.warn('%s is not a state variable.' % var, UserWarning)

    def is_initialized(self):
        return all(hasattr(self, var) for var in self._STATE_VARS)

    def fit(self, data, n_itns=1000, schedule={}, verbose=False, **kwargs):
        self._init_data(data)
        schedule = self._init_schedule(schedule)
        if not self.is_initialized():
            self._init_state_vars()
        assert self.is_initialized()
        self._check_params()
        self._update(data, n_itns=n_itns, schedule=schedule, verbose=verbose, **kwargs)
        return self

    def geweke_test(self, n_samples=10000, mask_type=None, n_itns=10, schedule={}, verbose=False, **kwargs):
        schedule = self._init_schedule(schedule)
        geweke(self, n_samples, mask_type, n_itns, schedule, verbose, **kwargs)

    def get_schedule(self, schedule={}):
        schedule = defaultdict(int, schedule)
        for k, v in schedule.items():
            if v is None:
                schedule[k] = np.inf
        return schedule

    def _init_schedule(self, schedule={}):
        schedule = defaultdict(int, schedule)
        for k, v in schedule.items():
            if v is None:
                schedule[k] = np.inf
            if k not in self._STATE_VARS:
                warnings.warn('%s is not a state variable.' % k, UserWarning)
        return schedule

    def _check_params(self):
        for s in self._STATE_VARS:
            if not np.isfinite(getattr(self, s)).all():
                raise ValueError('NaN/inf values in %s.' % s)

    def _get_vars_to_update(self, schedule):
        return [v for v in self._STATE_VARS if schedule[v] <= self._total_itns]

    def _update(self, data, n_itns=1000, schedule={}, verbose=False, **kwargs):
        schedule = self.get_schedule(schedule=schedule)
        for itn in xrange(n_itns):
            start_time = time.time()
            vars_to_update = self._get_vars_to_update(schedule)
            for var in vars_to_update:
                if verbose:
                    with timeit_context('updating %s' % var):
                        getattr(self, '_update_%s' % var)()
                        if hasattr(self, '_cache_%s' % var):
                            getattr(self, '_cache_%s' % var)()
                else:
                    getattr(self, '_update_%s' % var)()
                    if hasattr(self, '_cache_%s' % var):
                        getattr(self, '_cache_%s' % var)()
            if verbose:
                iter_time = time.time() - start_time
                print 'ITERATION %d:\t TOTAL TIME: %f' % (self._total_itns, iter_time)
            self._total_itns += 1


@contextmanager
def timeit_context(name):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    print '%3.4fs: %s' % (elapsedTime, name)


def get_state_vars(model):
    methods = [k for (k, v) in inspect.getmembers(model, predicate=inspect.ismethod)]
    update_methods = filter(lambda x: '_update_' in x, methods)
    return [x.split('_update_')[1] for x in update_methods]
