#!/usr/bin/env python3
"""
Simulate fast and slow synaptic plasticity in a teacher-student task. Need to
specify the type of plasticity rule ('Classical' or 'Bayesian'), type of
feedback ('cts' or 'spike'), and a random seed. Other customisable options can
be found below and in ../SynControl/parameters.py

Usage:
    python ./run_task1.py --argname1 arg1 --argname2 arg2 ...

"""

import argparse
import numpy as np
import pickle

from syn_control import bayes_learn
from syn_control import tasks
from syn_control import parameters
from syn_control import shared

parser = argparse.ArgumentParser(description='SynControl Task 1')
parser.add_argument('--model', type=str, default='Bayes', required=True,
                    help='Model type: Classical or Bayes')
parser.add_argument('--ftype', type=str, default='cts', required=True,
                    help='Feedback type: cts or spike')
parser.add_argument('--seed', type=int, default=0,  required=True,
                    help='Seed')
# optional arguments
parser.add_argument('--id', type=str, default='',
                    help='other info or index to param list')
parser.add_argument('--lag', type=int, default=0,
                    help='Feedback delay (ms)')
parser.add_argument('--gamma', type=float, default=1e-6,
                    help='Learning rate (Classical)')
parser.add_argument('--lam_u', type=float, default=0.1,
                    help='Control cost (Bayes)')
parser.add_argument('--sw_flag', type=int, default=1,
                    help='Plastic slow weights')
parser.add_argument('--fw_flag', type=int, default=1,
                    help='Plastic fast weights')
parser.add_argument('--save', type=bool,
                    help='Save results flag')
parser.add_argument('--save_steps', type=int, default=1,
                    help='Time increments for saving weights')
parser.add_argument('--path', type=str, default='../outputs/',
                    help='Save path')

args = parser.parse_args()
p = parameters.init_params()

for name in vars(args).keys():
    p.__setattr__(name, args.__getattribute__(name))


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
    xi = rng.standard_normal
    binom = rng.binomial

    task = tasks.ConstRates(p)
    w_tar, m, s2 = shared.weight_init(p)

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
    w_slow = np.array([wt.m for wt in weights])

    y_rmse = 0.  # will accrue RMS output error
    w_slow_rmse = [np.mean((w_slow.flatten() - w_tar.flatten())**2)**0.5]
    dw_traj = []  # these store fast, slow and target weight trajectories
    w_slow_traj = []
    w_tar_traj = []

    I_tar, r_tar = np.zeros((2, p.M))  # target current and firing rate
    y_tar = np.zeros(1)  # target output
    I, r = np.zeros((2, p.M))  # actual current and firing rate
    y = np.zeros(1)  # actual output
    f = np.zeros(p.M)  # feedback variable
    F = [np.zeros(p.M) for _ in range(int(p.lag/p.dt)+1)]  # buffer for delay
    for k in range(1, p.n_steps + 1):
        x_k = binom(1, task.rates*p.dt)
        noise = shared.noise(p, xi)

        # update model variables in place
        shared.evolve_model(w_tar, x_k, I_tar, r_tar, y_tar, p)
        shared.evolve_model(w, x_k, I, r, y, p, noise)
        shared.evolve_w(w_tar, p, xi)

        F.append(f)
        F.pop(0)
        if p.pop_control == 'global':
            gloErr.update(F[0])
            u_star = gloErr.u
        else:
            u_star = 0.
        for ind_m in range(p.M):
            weights[ind_m].update(x_k[ind_m], F[0][ind_m], u_star)

        f = fb(y - y_tar)

        y_rmse += (y - y_tar)**2
        w = np.array([wt.w for wt in weights])

        if k % int(p.save_steps/p.dt) == 0:
            w_slow = np.array([wt.m for wt in weights])
            w_slow_rmse.append(np.mean((w_slow.flatten() - w_tar.flatten())**2)**0.5)
            dw_traj.append(np.array([wt.dw for wt in weights]))
            w_slow_traj.append(w_slow.copy())
            w_tar_traj.append(w_tar.copy().flatten())

    y_rmse = (y_rmse/p.n_steps)**0.5
    w_slow_traj = np.array(w_slow_traj)
    dw_traj = np.array(dw_traj)
    w_tar_traj = np.array(w_tar_traj)
    print(y_rmse)
    results = {'params': p, 'W_err': w_slow_rmse, 'W_slow': w_slow_traj,
               'W_tar': w_tar_traj, 'dW': dw_traj, 'input_rates': task.rates,
               'Y_rmse': y_rmse}
    return results


if __name__ == '__main__':
    results = run(p)
    if p.save:
        filename = (f'task1_{p.model}_{p.ftype}_{p.id}_seed{p.seed}')
        pickle.dump(results, open(f'{p.path}{filename}', 'wb'))
