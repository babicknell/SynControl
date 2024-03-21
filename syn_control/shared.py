import numpy as np


def evolve_model(w, x, I, r, y, p, noise=(0., 0., 0.)):
    """ Evolve model dynamics

    Parameters
    ----------
    w : ndarray
        synaptic weights
    x : ndarray
        synaptic input
    I, r, y: ndarray
        current, rate and output variables
    p : data class instance
        model and simulation parameters
    noise : tuple
        white noise input

    """
    I += -p.dt/p.tau_I*I + 1/p.tau_I*np.sum(w*x, -1) + noise[0]
    r += -p.dt/p.tau_r*r + p.alpha_Ir*p.dt/p.tau_r*I + noise[1]
    y += -p.dt/p.tau_y*y + p.dt/p.tau_y*np.sum(r) + noise[2]


def evolve_w(w_tar, p, xi):
    """ Evolve target weights via OU process

    Parameters:
    ----------
    w_tar : ndarray
        target synaptic weights
    p : data class instance
        model and simulation parameters
    xi : function
        rng.standard_normal

    """
    w_tar += (
            -p.dt / p.tau_w * (w_tar - p.m_prior) +
            np.sqrt(2 * p.dt * p.s2_prior / p.tau_w) * xi(p.num_syns)
            )


def noise(p, xi):
    """
    Generate noise for model dynamics
    Parameters
    ----------
    p : data class instance
        model and simulation parameters
    xi : function
        rng.standard_normal

    Returns
    -------
    tuple : model noise

    """
    return (np.sqrt(2*p.dt*p.sig2_I/p.tau_I)*xi(),
               np.sqrt(2*p.dt*p.sig2_r/p.tau_r)*xi(),
               np.sqrt(2*p.dt*p.sig2_y/p.tau_y)*xi())


def feedbackC(delta, p, xi):
    """
    Generate continuous-valued error feedback, delta + white noise
    
    Parameters
    ----------
    delta : float
        output error, delta = y - y*
    p : data class instance
        model and simulation parameters
    xi : function
        rng.standard_normal

    Returns
    -------
    ndarray : feedback signal for M neurons
    
    """
    return delta + np.sqrt(p.sig2_f/p.dt)*xi(p.M)


def feedbackS(delta, p, binom):
    """
    Generate spike-base error feedback (Poisson process)

    Parameters
    ----------
    delta : float
        output error, delta = y - y*
    p : data class instance
        model and simulation parameters
    binom : function
        rng.binomial

    Returns
    -------
    ndarray : feedback signal for M neurons

    """
    spike_prob = np.min([1e-3*p.dt*p.eta0*np.exp(p.rho*delta).squeeze(), 1.])
    return binom(1, spike_prob, size=p.M)


def weight_init(p):
    """
    Initialise synaptic weights

    Parameters
    ----------
    p : data class instance
        model and simulation parameters

    Returns
    -------
    w_tar: ndarray
        target weights
    m : ndarray
        actual weights (mean)
    s2 : ndarray
        actual weights (variance)

    """
    rng = np.random.default_rng(seed=p.seed)
    w_tar = rng.normal(p.m_prior, p.s2_prior**0.5, (p.M, p.num_syns))
    m = rng.normal(p.m_prior, p.s2_prior**0.5, (p.M, p.num_syns))
    s2 = p.s2_prior*np.ones((p.M, p.num_syns))
    return w_tar, m, s2


def smooth(x, width):
    """ smooth input x with moving average of specified width """
    return np.convolve(x, 1/width*np.ones(width), mode='valid')
