import numpy as np


class ConstRates:
    """
    For defining constant input rates to synapses

    Parameters
    ----------
    p : data class instance
        model and simulation parameters

    Attributes
    ----------
    p : data class instance
        model and simulation parameters
    rng : random_generator instance
        np.random.rng(seed=seed)
    rates: ndarray
        synaptic input rates for M neurons X num_syn synapses

    """
    def __init__(self, p):
        self.p = p
        self.rng = np.random.default_rng(seed=p.seed)
        self.rates = self.rng.uniform(0, 2*p.nu_bar, size=(p.M, p.num_syns))
        self.rates[self.rates*p.dt > 1] = 1/p.dt

    def update(self, step):
        pass


class FourierCurve:
    """
    For generating and evolving target output trajectories from random Fourier
    series

    Parameters
    ----------
    p : data class instance
        model and simulation parameters
    seed : int
        random seed

    Attributes
    ----------
    p : data class instance
        model and simulation parameters
    rng : random_generator instance
        np.random.rng(seed=seed)
    A, B: ndarray
        Fourier coefficients
    phi: ndarray
        bump function for setting boundaries
    amp: int
        scales amplitude of curve
    step: int
        current (periodic) time step
    Y_period: ndarray
        curve evaluated over whole period
    Y: float
        curve evaluated at current step

    """

    def __init__(self, p, seed):
        self.p = p
        self.rng = np.random.default_rng(seed=seed)
        self.A = self.rng.normal(0, (1/(2*self.p.Ymodes))**0.5, self.p.Ymodes)
        self.B = self.rng.normal(0, (1/(2*self.p.Ymodes))**0.5, self.p.Ymodes)
        tt = np.linspace(-1, 1, int(self.p.period/self.p.dt))
        self.phi = self.bump(tt, 1)
        self.amp = 10
        self.step = 0
        self.Y_period = self.make_curve()
        self.Y = self.Y_period[self.step]
        self.update()

    def bump(self, x, rho):
        """ Bump function with input x and steepness parameter rho. """
        return np.exp(-rho/(1e-8 + 1-x**2))/np.exp(-rho/(1e-8 + 1))

    def make_curve(self):
        """ Sum over Fourier components, multiply by bump function and
        add mean value. """
        Y_p = np.zeros(int(self.p.period/self.p.dt))
        for t in range(int(self.p.period/self.p.dt)):
            Y_p[t] = np.sum([self.A[k]*np.cos(2*np.pi*k*t*self.p.dt/self.p.period) +
                             self.B[k]*np.sin(2*np.pi*k*t*self.p.dt/self.p.period)
                             for k in range(self.p.Ymodes)])

        Y_p = self.amp*Y_p * self.phi + self.p.Ymean
        return Y_p

    def update_coeffs(self):
        """ Evolve Fourier coefficients by OU process. """
        self.A += (-self.p.dt/self.p.tau_w*self.A +
                   np.sqrt(1/(self.p.Ymodes)*self.p.dt/self.p.tau_w) *
                   self.rng.standard_normal(self.p.Ymodes)
                   )
        self.B += (-self.p.dt/self.p.tau_w*self.B +
                   np.sqrt(1/(self.p.Ymodes)*self.p.dt/self.p.tau_w) *
                   self.rng.standard_normal(self.p.Ymodes)
                   )

    def update(self):
        """ Update current value of curve and increment step. """
        self.Y = np.sum([self.A[k]*np.cos(2*np.pi*k*self.step*self.p.dt/self.p.period) +
                         self.B[k]*np.sin(2*np.pi*k*self.step*self.p.dt/self.p.period)
                         for k in range(self.p.Ymodes)])
        self.Y = self.amp*self.Y * self.phi[self.step] + self.p.Ymean
        self.step = int((self.step + 1) % (self.p.period/self.p.dt))


class GaussBumpRates:
    """
    For generating Gaussian bumps of pre-synaptic firing rates

    Parameters
    ----------
    p : data class instance
        model and simulation parameters
    seed : int
        random seed

    Attributes
    ----------
    p : data class instance
        model and simulation parameters
    rng : random_generator instance
        np.random.rng(seed=seed)
    inds: ndarray
        indices of randomly chosen active synapses
    shift: ndarray
        random periodic shifts to set Gaussian centres for different synapses
    amp: ndarray
        amplitudes of Gaussian bumps, with random jitter
    step: int
        current (periodic) time step
    R: ndarray
        template Gaussian bump curve
    rates: ndarray
        firing rates at current time step


    """
    def __init__(self, p, seed):
        self.rng = np.random.default_rng(seed=seed)
        self.p = p
        self.inds = self.rng.choice(np.arange(p.num_syns),
                                    size=int(p.frac*p.num_syns), replace=False)
        self.shift = self.rng.uniform(-(p.period/p.dt)/2, (p.period/p.dt)/2,
                                 len(self.inds))
        self.amp = 1e-3*(p.input_amp + 0.2*p.input_amp*(self.rng.random(int(p.frac*p.num_syns)) - 0.5))
        self.step = 0
        self.R = self.gaussian_bump()
        self.rates = np.zeros(self.p.num_syns)
        self.update()

    def gaussian_bump(self):
        """ Make template curve with specified bump width. """
        R = np.zeros(int(self.p.period/self.p.dt))
        mu = self.p.period/2
        sig = self.p.input_width
        for t in range(len(R)):
            R[t] = np.exp(-1/(2*sig**2)*(self.p.dt*t - mu)**2)
        return R

    def update(self):
        """ Update input rates to active synapses and increment step """
        active = self.amp*np.array([self.R[int((self.step + self.shift[k]) %
                                (self.p.period/self.p.dt))] for
                                k in range(len(self.inds))])
        self.rates[self.inds] = active
        self.step = (self.step + 1) % (self.p.period/self.p.dt)

