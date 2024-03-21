#!/usr/bin/env python3
"""
Simulate fast and slow synaptic plasticity in a toy linear regression model,
save results to outputs directory.

Usage:
    python ./run_simple_model.py --argname1 arg1 --argname2 arg2 ...

"""

import argparse
import numpy as np
import pickle

from matplotlib import pyplot as plt


parser = argparse.ArgumentParser(
    description='Simple Model')
parser.add_argument('--model', type=str, default='FastSlow',
                    help='Model type: Classical or FastSlow')
parser.add_argument('--n_syns', type=int, default='20',
                    help='number of synapses')
parser.add_argument('--T', type=float, default='1e4',
                    help='Total time (ms)')
parser.add_argument('--dt', type=float, default=1.,
                    help='Time step (ms)')
parser.add_argument('--period', type=int, default=1000,
                    help='Stimulus period (ms)')
parser.add_argument('--seed', type=int, default=0, help='Seed')
parser.add_argument('--nu_bar', type=float, default=1.,
                    help='Average input rate')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--save', type=bool, default=False,
                    help='Save results flag')
parser.add_argument('--save_steps', type=int, default=1,
                    help='Time increments for saving weights')
parser.add_argument('--path', type=str, default='../outputs/',
                    help='Save path')
parser.add_argument('--plot', type=bool, default=True,
                    help='plot flag')

p = parser.parse_args()


def inputs(t, p):
    """
    Generate time-varying synaptic input rates from Fourier basis functions.

    Parameters
    ----------
    t :  float
        time
    p : data class instance
        model and simulation parameters

    Returns
    -------
    ndarray : synaptic input rates

    """
    Xc = [np.cos(2*np.pi*k*t*p.dt/p.period) for k in range(1, p.n_syns//2+1)]
    Xs = [np.sin(2*np.pi*k*t*p.dt/p.period) for k in range(1, p.n_syns//2+1)]
    return p.nu_bar + np.array(Xc + Xs)


def run(p):
    """
    Run simulation.

    Parameters
    ----------
    p : data class instance
        model and simulation parameters

    Returns
    -------
    results : dict
        dictionary of results (output, weights, etc.)

    """
    rng = np.random.default_rng(seed=p.seed)

    w_tar = rng.standard_normal(p.n_syns) / p.n_syns ** 0.5
    w0 = rng.standard_normal(p.n_syns) / p.n_syns ** 0.5
    dw0 = np.zeros(p.n_syns)

    w = w0.copy()
    dw = dw0.copy()
    Y = []
    Y_tar = []
    W = []
    dW = []
    for t in range(int(p.T/p.dt)):
        X = inputs(t, p)
        y_tar = w_tar@X
        y = (w + dw)@X
        f = y - y_tar

        w -= p.lr*(f - dw*p.n_syns*p.nu_bar)*X
        if p.model == 'FastSlow':
            dw -= f/(p.n_syns*p.nu_bar)

        Y.append(y)
        Y_tar.append(y_tar)
        W.append(w.copy())
        dW.append(dw.copy())

    Y = np.array(Y)
    Y_tar = np.array(Y_tar)
    W = np.array(W)
    dW = np.array(dW)

    results = {'Y_tar': Y_tar, 'Y': Y, 'W': W, 'dW': dW,
               'W_tar': w_tar, 'params': p}

    if p.plot:
        # plot output and target output vs time
        fig, ax = plt.subplots(1, 2, figsize=(8, 2.5))
        t = np.arange(int(p.T/p.dt))
        early = np.arange(int(1000/p.dt))
        late = np.arange(int((p.T-1000)/p.dt), int(p.T/p.dt))
        ax[0].plot(t[early], Y_tar[early], '--', color='k')
        ax[0].plot(t[early], Y[early], '-', color='C1')
        ax[1].plot(t[late], Y_tar[late], '--', color='k')
        ax[1].plot(t[late], Y[late], '-', color='C1')

        for a in ax:
            a.set_ylabel('output (a.u.)', fontsize=14)
            a.set_xlabel('time (s)', fontsize=14)
            a.tick_params(labelsize=14)
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
        ax[0].set_xticks([early[0], early[-1]+1])
        ax[0].set_xticklabels([int(1e-3*p.dt*early[0]), int(1e-3*p.dt*(early[-1] + 1))])
        ax[1].set_xticks([late[0], late[-1]+1])
        ax[1].set_xticklabels([int(1e-3 * p.dt * late[0]), int(1e-3 * p.dt * (late[-1] + 1))])
        ax[0].set_title('early', fontsize=14)
        ax[1].set_title('late', fontsize=14)
        ax[0].legend(['target', 'actual'])
        fig.tight_layout()

        # plot RMS weight error vs time
        fig2, ax2 = plt.subplots(figsize=(5, 2.5))

        w_err = np.mean((W - w_tar)**2, 1)**0.5
        ax2.plot(w_err)
        ax2.set_xlabel('time (s)', fontsize=14)
        ax2.set_ylabel('weight RMSE (a.u.)', fontsize=14)
        ax2.tick_params(labelsize=14)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_xticks([early[0], late[-1] + 1])
        ax2.set_xticklabels([int(1e-3 * p.dt * early[0]), int(1e-3 * p.dt * (late[-1] + 1))])
        ax2.set_yticks([0, np.round(np.max(w_err)+0.05, 1)])
        fig2.tight_layout()

        plt.show()

    return results


if __name__ == '__main__':
    results = run(p)
    if p.save:
        filename = f'simple_{p.model}_lr{p.lr}_seed{p.seed}'
        pickle.dump(results, open(f'{p.path}{filename}', 'wb'))
