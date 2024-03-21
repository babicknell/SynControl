#!/usr/bin/env python3
"""
Simulate fast and slow synaptic plasticity in a cerebellar learning task. Need to
specify the type of plasticity rule ('Classical' or 'Bayesian'), type of
feedback ('cts' or 'spike'), and a random seed. Other customisable options can
be found below and in ../SynControl/parameters.py

Usage:
    ./run_task2.py --argname1 arg1 --argname2 arg2 ...

"""

import argparse
import numpy as np
import pickle

from syn_control import bayes_learn
from syn_control import tasks
from syn_control import parameters
from syn_control import shared

parser = argparse.ArgumentParser(
    description='SynControl Task 2 (cerebellar learning)')
parser.add_argument('--model', type=str, default='Bayes',
                    help='Model type: Classical or Bayes', required=True)
parser.add_argument('--ftype', type=str, default='cts', required=True,
                    help='Feedback type: cts or spike')
parser.add_argument('--seed', type=int, default=0, required=True,
                    help='Seed')
# optional arguments
parser.add_argument('--id', type=str, default='',
                    help='other info or index to param list')
parser.add_argument('--lag', type=int, default=0,
                    help='Feedback delay (ms)')
parser.add_argument('--gamma', type=float, default=1e-6,
                    help='Learning rate (Classical)')
parser.add_argument('--lam_u', type=float, default=1.,
                    help='Control cost (Bayes)')
parser.add_argument('--sw_flag', type=int, default=1,
                    help='Plastic slow weights')
parser.add_argument('--fw_flag', type=int, default=1,
                    help='Plastic fast weights')
parser.add_argument('--save', type=bool,
                    help='Save results flag')
parser.add_argument('--save_steps', type=int, default=1,
                    help='Time increments for saving weights')
parser.add_argument('--checkpoint', type=int, default=100000,
                    help='Time increments for checkpointing')
parser.add_argument('--path', type=str, default='../outputs/task2/',
                    help='Save path')
# specific to task 2
parser.add_argument('--period', type=float, default=1e3,
                    help='Stimulus period (ms)')
parser.add_argument('--Ymean', type=float, default=20.,
                    help='Target output mean')
parser.add_argument('--Ymodes', type=int, default=3,
                    help='Number of target output Fourier modes')
parser.add_argument('--input_amp', type=float, default=100.,
                    help='Amplitude of input temporal tuning curve (ms)')
parser.add_argument('--input_width', type=float, default=100.,
                    help='Width of input temporal tuning curve (ms)')
parser.add_argument('--num_patterns', type=int, default=10,
                    help='Number of input-output pairs')
parser.add_argument('--frac', type=float, default=0.5,
                    help='Fraction of active synapses per pattern')
parser.add_argument('--max_seed', type=int, default=10,
                    help='Max allowable random seed')

args = parser.parse_args()
assert args.seed <= args.max_seed  # this is here to ensure that random input
# and output curves do not overlap across seeds (see Rates and targets below)

p = parameters.init_params()
for name in vars(args).keys():
    p.__setattr__(name, args.__getattribute__(name))

p.nu_bar = 1e-3*p.frac*p.input_amp*np.sqrt(2*np.pi)*p.input_width/p.period  #
# average input rate for Gaussian bumps of activity
p.filename = (f'task2_{p.model}_{p.ftype}_{p.id}_'
              f'seed{p.seed}')


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
    try:
        #  look for checkpointed file first, otherwise initialize from scratch
        cpt = pickle.load(open(f'{p.path}{p.filename}_checkpoint', 'rb'))
        (
        p, k, ind, r, I, y, F, X, mli, dw_traj, w_slow_traj,
        w_var_traj, y_rmse, weights, Rates, targets, rng, xi, binom, fb
        ) = (
             cpt['p'], cpt['k'], cpt['ind'], cpt['r'], cpt['I'], cpt['y'],
             cpt['F'], cpt['X'], cpt['gloErr'], cpt['dw_traj'],
             cpt['w_slow_traj'], cpt['w_var_traj'], cpt['y_rmse'],
             cpt['weights'], cpt['Rates'], cpt['targets'], cpt['rng'],
             cpt['xi'], cpt['binom'], cpt['fb']
        )
        w = np.array([wt.w for wt in weights])

    except FileNotFoundError:

        rng = np.random.default_rng(seed=p.seed)
        xi = rng.standard_normal
        binom = rng.binomial

        Rates = [[tasks.GaussBumpRates(p, np.ravel_multi_index([p.seed, m, k],
                (p.max_seed, p.M, p.num_patterns))) for k in range(p.num_patterns)]
                 for m in range(p.M)]  # input rates to M X num_syn synapses
        targets = [tasks.FourierCurve(p, np.ravel_multi_index([p.seed, k],
                (p.max_seed, p.num_patterns))) for k in range(p.num_patterns)]
        _, m, s2 = shared.weight_init(p)

        if p.ftype == 'cts':

            def fb(delta):
                return shared.feedbackC(delta, p, xi)

            if p.model == 'Bayes':
                weights = [bayes_learn.BayesLearner(p, m[k], s2[k]) for k in
                           range(p.M)]
            else:
                weights = [bayes_learn.ClasLearner(p, m[k]) for k in
                           range(p.M)]

            if p.pop_control == 'global':
                gloErr = bayes_learn.ErrorEst(p)

        elif p.ftype == 'spike':

            def fb(delta):
                return shared.feedbackS(delta, p, binom)

            if p.model == 'Bayes':
                weights = [bayes_learn.BayesLearnerS(p, m[k], s2[k]) for k in
                           range(p.M)]
            else:
                weights = [bayes_learn.ClasLearnerS(p, m[k]) for k in
                           range(p.M)]

            if p.pop_control == 'global':
                gloErr = bayes_learn.ErrorEstS(p)

        else:
            print('unknown feedback type')
            exit()

        w = np.array([wt.w for wt in weights])

        # simulation
        y_rmse = 0.  # will accrue RMS output error
        dw_traj = []  # fast weight trajectory
        w_slow_traj = []  # slow weight trajectory
        w_var_traj = []  # slow weight variance trajectory

        I = np.zeros(p.M)  # total input current
        r = np.zeros(p.M)  # firing rate
        y = np.array([0.])  # output
        f = np.zeros(p.M)  # feedback variable
        F = [np.zeros(p.M) for _ in range(int(p.lag/p.dt) + 1)]  #buffer for delay
        ind = 0  # index for current presented I/O patttern
        k = 1  # time step

    while k < (p.n_steps + 1):
        x_k = np.array([binom(1, p.dt * Rates[m][ind].rates) for m in range(p.M)])
        noise = shared.noise(p, xi)
        shared.evolve_model(w, x_k, I, r, y, p, noise)

        for m in range(p.M):
            Rates[m][ind].update()  # evolve input rates in time

        for ii in range(p.num_patterns):
            targets[ii].update_coeffs()   # evolve target outputs in time

        targets[ind].update()
        if k % (p.period/p.dt) == 0:
            # randomly choose new I/O pattern at each period
            ind = rng.integers(0, p.num_patterns)

        F.append(f)
        F.pop(0)
        if p.pop_control == 'global':
            gloErr.update(F[0])
            u_star = gloErr.u
        else:
            u_star = 0.
        for ind_m in range(p.M):
            weights[ind_m].update(x_k[ind_m], F[0][ind_m], u_star)

        y_tar = targets[ind].Y
        f = fb(y - y_tar)

        y_rmse += (y - y_tar) ** 2
        w = np.array([wt.w for wt in weights])

        if k % int(p.save_steps / p.dt) == 0:
            dw_traj.append(np.array([wt.dw for wt in weights]))
            w_slow_traj.append(np.array([wt.m for wt in weights]))
            w_var_traj.append(np.array([wt.s2 for wt in weights]))

        if k % int(p.checkpoint / p.dt) == 0:
            cpt = {'p': p, 'k': k + 1, 'ind': ind, 'r': r, 'I': I, 'y': y,
                   'F': F, 'X': X, 'gloErr': gloErr,
                   'dw_traj': dw_traj, 'w_slow_traj': w_slow_traj,
                   'w_var_traj': w_var_traj, 'y_rmse': y_rmse,
                   'weights': weights, 'Rates': Rates, 'targets': targets,
                   'rng': rng, 'xi': xi, 'binom': binom, 'fb': fb}
            pickle.dump(cpt, open(f'{p.path}{p.filename}_checkpoint', 'wb'))
            print(int(p.dt * k))
        k += 1

    y_rmse = (y_rmse / p.n_steps) ** 0.5
    w_slow_traj = np.array(w_slow_traj)
    dw_traj = np.array(dw_traj)
    w_var_traj = np.array(w_var_traj)
    print(y_rmse)
    results = {'params': p, 'W_slow': w_slow_traj, 'W_var': w_var_traj,
               'dW': dw_traj, 'input_rates': Rates, 'targets': targets,
               'Y_rmse': y_rmse}
    return results


if __name__ == '__main__':
    results = run(p)
    if args.save:
        pickle.dump(results, open(f'{p.path}{p.filename}', 'wb'))
