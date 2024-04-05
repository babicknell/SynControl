import numpy as np
from scipy import linalg
import copy


class BayesLearner:
    """
    Bayesian synapse object (continuous feedback version). Uses a generative
    model of error feedback to optimally update synaptic weights on
    slow (learning) and fast (control) timescales.

    Parameters
    ----------
    p : data class instance
        model and simulation parameters
    m :  ndarray
        initial weights (mean)
    s2 : ndarray
        initial weights (variance)

    Attributes
    ----------
    p : data class instance
        model and simulation parameters
    m :  ndarray
        slow weights (mean)
    s2 : ndarray
        slow weights (variance)
    dw : ndarray
        fast weight variable
    w : ndarray
        total weight variable, m + dw
    h: ndarray
        convenience vector for use in plasticity rule
    lam : ndarray
        covariance matrix of error dynamics noise
    expA_lag : ndarray
        operator to propagate lagged predicted error forward in time
    B :  ndarray
        convenience matrix for computing control gain
    L : ndarray
        control gain
    K : ndarray
        Kalman gain
    Sig2 : ndarray
        Error estimate error-covariance matrix
    d :  ndarray
        error estimate
    d_pred : ndarray
        error prediction (when propagted to handle delays)
    z :  ndarray
        eligibility trace
    U : ndarray
        leaky integral of past controls
    u_lgd : list
        buffer for maintaining lagged control variable
    X : list
        buffer for maintaining lagged synaptic input
    input : ndarray
        synaptic input scaled by weight variance

    """
    def __init__(self, p, m, s2):
        self.p = copy.deepcopy(p)
        self.m = m
        self.s2 = s2
        self.dw = np.array(0.)
        self.w = self.m + self.dw

        self.h = np.array([0, 0, 1]).reshape(-1, 1)
        self.lam = np.diag([p.M*p.sig2_0, 2*p.M/p.tau_r*p.sig2_r,
                            2/p.tau_y*p.sig2_y])
        self.expA_lag = linalg.expm(p.A*p.lag)
        _, self.L, self.B = Ricatti_ss(p)
        self.Sig2, self.K = Kalman_ss(p, self.lam)

        self.d = np.zeros((3, 1))
        self.d_pred = np.zeros((3, 1))
        self.z = np.zeros((3, p.num_syns))
        self.U = np.zeros((3, 1))
        self.u_lgd = [np.zeros((3, 1)) for _ in range(int(p.lag/p.dt)+1)]
        self.X = [np.zeros((p.M, p.num_syns)) for _ in range(int(p.lag/p.dt)+1)]
        self.input = np.zeros((3, p.num_syns))

    def update(self, x, f):
        """
        Integrates input and feedback, then updates synaptic weights.

        Parameters
        ----------
        x : ndarray
            synaptic input
        f : float
            error feedback
        u : float
            external control variable (zero unless p.pop_control == 'global')


        """

        self.X.append(x)
        self.X.pop(0)
        self.input[0] = 1/self.p.tau_I*self.s2*self.X[0]

        self.d += self.p.dt*(
                self.p.A@self.d + self.u_lgd[0] +
                self.K*(f - self.d[-1])
                )

        self.z += (
                self.p.dt*(self.p.A@self.z - self.K*self.z[-1]) + self.input
                )

        if self.p.sw_flag:
            self.m += self.p.dt*(
                    -1/self.p.sig2_f*self.z[-1]*(f - self.d[-1]) -
                    1/self.p.tau_w*(self.m - self.p.m_prior)
                    )
            self.s2 += self.p.dt*(
                    -1/self.p.sig2_f*self.z[-1]**2 -
                    2/self.p.tau_w*(self.s2 - self.p.s2_prior)
                    )
            self.s2[self.s2 < 0] = 0.

        if self.p.fw_flag:
            self.d_pred = self.expA_lag@self.d + self.U
            self.dw = -1/(self.p.M*self.p.num_syns*self.p.nu_bar)*(
                    self.L@self.d_pred)[0].squeeze()

            self.u_lgd.append(-self.B@self.L@self.d_pred)
            self.u_lgd.pop(0)
            self.U += self.p.dt*(
                    self.p.A@self.U - self.expA_lag@self.u_lgd[0] +
                    self.u_lgd[-1]
                    )

        self.w = self.m + self.p.fw_flag*self.dw


class BayesLearnerApprox:
    """
    Bayesian synapse object (approximated continuous feedback version). Uses a
    generative model of error feedback to optimally update synaptic weights on
    slow (learning) and fast (control) timescales.

    Parameters
    ----------
    p : data class instance
        model and simulation parameters
    m :  ndarray
        initial weights (mean)
    s2 : ndarray
        initial weights (variance)

    Attributes
    ----------
    p : data class instance
        model and simulation parameters
    m :  ndarray
        slow weights (mean)
    s2 : ndarray
        slow weights (variance)
    dw : ndarray
        fast weight variable
    w : ndarray
        total weight variable, m + dw
    sig2_d : float
        covariance of error dynamics noise
    expA_lag : float
        operator to propagate lagged predicted error forward in time
    B : float
        convenience parameter for computing control gain
    L : float
        control gain
    K : float
        Kalman gain
    d :  float
        error estimate
    d_pred : float
        error prediction (when propagated to handle delays)
    z :  ndarray
        eligibility trace
    U : float
        leaky integral of past controls
    u_lgd : list
        buffer for maintaining lagged control variable
    X : list
        buffer for maintaining lagged synaptic input
    xr : ndarray
        low-pass filtered input

    """
    def __init__(self, p, m, s2):
        self.p = copy.deepcopy(p)
        self.m = m
        self.s2 = s2
        self.dw = np.array(0.)
        self.w = self.m + self.dw

        self.sig2_d = (
                p.alpha_Ir**2*p.tau_I**2*p.sig2_0 + 2*p.tau_r*p.sig2_r +
                2*p.tau_y*p.sig2_y)
        self.expA_lag = np.exp(-p.lag/p.tau_y)
        self.B = p.alpha_Ir/p.tau_y
        self.L = 1/p.lam_u**0.5
        self.K = (self.sig2_d/p.sig2_f)**0.5

        self.d = 0.
        self.d_pred = 0.
        self.z = np.zeros(p.num_syns)
        self.U = 0.
        self.u_lgd = [0. for _ in range(int(p.lag/p.dt)+1)]
        self.X = [np.zeros((p.M, p.num_syns)) for _ in range(int(p.lag/p.dt)+1)]
        self.xr = np.zeros(p.num_syns)

    def update(self, x, f):
        """
        Integrates input and feedback, then updates synaptic weights.

        Parameters
        ----------
        x : ndarray
            synaptic input
        f : float
            error feedback

        """
        self.X.append(x)
        self.X.pop(0)
        self.xr += (
                -self.p.dt/self.p.tau_r*self.xr +
                1/self.p.tau_r*self.X[0].squeeze()
                )

        self.d += self.p.dt*(
                -1/self.p.tau_y*self.d + self.u_lgd[0] +
                1/self.p.tau_y*self.K*(f - self.d)
                )

        self.z += self.p.dt*(
                -1/self.p.tau_y*self.z - 1/self.p.tau_y*self.K*self.z +
                self.p.alpha_Ir/self.p.tau_y*self.xr*self.s2
                )

        if self.p.sw_flag:
            self.m += self.p.dt*(-1/self.p.sig2_f*self.z*(f - self.d) -
                                  1/self.p.tau_w*(self.m - self.p.m_prior)
                                  )
            self.s2 += self.p.dt*(-1/self.p.sig2_f*(self.z)**2 -
                                   2/self.p.tau_w*(self.s2 - self.p.s2_prior)
                                   )
            self.s2[self.s2 < 0] = 0.

        if self.p.fw_flag:
            self.d_pred = self.expA_lag*self.d + self.U
            self.dw = -1/(self.p.num_syns*self.p.nu_bar)*self.L*self.d_pred
            self.u_lgd.append(-self.B*self.L*self.d_pred)
            self.u_lgd.pop(0)
            self.U += self.p.dt*(-1/self.p.tau_y*self.U -
                                 self.expA_lag*self.u_lgd[0] + self.u_lgd[-1])

        self.w = self.m + self.p.fw_flag*self.dw


class BayesLearnerS:
    """
    Bayesian synapse object (spiking feedback version). Uses a generative
    model of error feedback to optimally update synaptic weights on
    slow (learning) and fast (control) timescales.

    Parameters
    ----------
    p : data class instance
        model and simulation parameters
    m :  ndarray
        initial weights (mean)
    s2 : ndarray
        initial weights (variance)

    Attributes
    ----------
    p : data class instance
        model and simulation parameters
    m : ndarray
        slow weights (mean)
    s2 : ndarray
        slow weights (variance)
    dw : ndarray
        fast weight variable
    w : ndarray
        total weight variable, m + dw
    h: ndarray
        convenience vector for use in plasticity rule
    lam : ndarray
        covariance matrix of error dynamics noise
    expA_lag : ndarray
        operator to propagate lagged predicted error forward in time
    eps : float
        a small number
    B :  ndarray
        convenience matrix for computing control gain
    L : ndarray
        control gain
    Sig2 : ndarray
        error estimate error-covariance matrix
    q : ndarray
        estimate of feedback rate
    d :  ndarray
        error estimate
    d_pred : ndarray
        error prediction (when propagated to handle delays)
    z :  ndarray
        eligibility trace
    U : ndarray
        leaky integral of past controls
    u_lgd : list
        buffer for maintaining lagged control variable
    X : list
        buffer for maintaining lagged synaptic input
    input : ndarray
        synaptic input scaled by weight variance

    """
    def __init__(self, p, m, s2):
        self.p = copy.deepcopy(p)
        self.m = m
        self.s2 = s2
        self.dw = np.array(0.)
        self.w = self.m + self.dw

        self.h = np.array([0, 0, 1]).reshape(-1, 1)
        self.lam = np.diag([p.M*p.sig2_0, 2*p.M/p.tau_r*p.sig2_r,
                            2/p.tau_y*p.sig2_y])
        self.expA_lag = linalg.expm(p.A*p.lag)
        self.eps = 1e-16
        _, self.L, self.B = Ricatti_ss(p)
        self.Sig2 = self.lam.copy()

        self.q = np.array(0.)
        self.d = np.zeros((3, 1))
        self.d_pred = np.zeros((3, 1))
        self.z = np.zeros((3, p.num_syns))
        self.U = np.zeros((3, 1))
        self.u_lgd = [np.zeros((3, 1)) for _ in range(int(p.lag/p.dt)+1)]
        self.X = [np.zeros((p.M, p.num_syns)) for _ in range(int(p.lag/p.dt)+1)]
        self.input = np.zeros((3, self.p.num_syns))

    def update(self, x, f):
        """
        Integrates input and feedback, then updates synaptic weights.

        Parameters
        ----------
        x : ndarray
            synaptic input
        f : float
            error feedback
        u : float
            external control variable (zero unless p.pop_control == 'global')

        """
        self.X.append(x)
        self.X.pop(0)
        self.input[0] = 1./self.p.tau_I*self.X[0]*self.s2

        self.q = min(1e-3*self.p.dt*self.p.eta0*np.exp(self.p.rho*self.d[-1] +
                    1/2*self.p.rho**2*self.Sig2[-1, -1]), 1. - self.eps)

        self.d += (
                self.p.dt*(self.p.A@self.d + self.u_lgd[0]) +
                self.p.rho*self.Sig2@self.h*(f - self.q)
                )

        self.Sig2 += (
                self.p.dt*(self.p.A@self.Sig2 + self.Sig2@self.p.A.T + self.lam) -
                self.p.rho**2*self.q*self.Sig2@self.h@self.h.T@self.Sig2
                )

        self.z += (
                self.p.dt*self.p.A@self.z -
                self.p.rho**2*self.q*self.Sig2@self.h@self.h.T@self.z +
                self.input
                )

        if self.p.sw_flag:
            self.m += (
                    -self.p.rho*(f - self.q)*self.z[-1] -
                    self.p.dt/self.p.tau_w*(self.m - self.p.m_prior)
                    )
            self.s2 += (
                    -self.p.rho**2*self.q*self.z[-1]**2 -
                    2*self.p.dt/self.p.tau_w*(self.s2 - self.p.s2_prior)
                    )
            self.s2[self.s2 < 0] = 0.

        if self.p.fw_flag:
            self.d_pred = self.expA_lag@self.d + self.U
            self.dw = -1./((1 + self.p.beta*(self.p.M - 1))*
                           self.p.num_syns*self.p.nu_bar)*(
                    self.L@self.d_pred)[0].squeeze()

            self.u_lgd.append(-self.B@self.L@self.d_pred)
            self.u_lgd.pop(0)
            self.U += self.p.dt*(
                    self.p.A@self.U - self.expA_lag@self.u_lgd[0] +
                    self.u_lgd[-1]
                    )
        self.w = self.m + self.p.fw_flag*self.dw


class ClasLearner:
    """
    Classical synapse object (continuous feedback version). Uses online gradient
    descent on MSE loss to update synaptic weights.

    Parameters
    ----------
    p : data class instance
        model and simulation parameters
    w : ndarray
        initial weights

    Attributes
    ----------
    p : data class instance
        model and simulation parameters
    m : ndarray
        dummy slow weights (mean)
    s2 : ndarray
        initial slow weights (variance; unused)
    dw : ndarray
         dummy fast weight variable (unused)
    w : ndarray
        weight variable
    h: ndarray
        convenience vector for use in plasticity rule
    z :  ndarray
        eligibility trace
    X : list
        buffer for maintaining lagged synaptic input
    input : ndarray
        padded synaptic input, scaled by time constant

    """
    def __init__(self, p, w):
        self.p = p
        self.m = w
        self.w = w
        self.s2 = []
        self.dw = np.array(0.)

        self.z = np.zeros((3, p.num_syns))
        self.X = [np.zeros((p.M, p.num_syns)) for _ in range(int(p.lag/p.dt)+1)]
        self.input = np.zeros((3, p.num_syns))

        self.h = np.array([0, 0, 1]).reshape(-1, 1)

    def update(self, x, f):
        """
        Integrates input and feedback, then updates synaptic weights.

        Parameters
        ----------
        x : ndarray
            synaptic input
        f : float
            error feedback

        """
        self.X.append(x)
        self.X.pop(0)
        self.input[0] = 1/self.p.tau_I*self.X[0]

        self.z += self.p.dt*self.p.A@self.z + self.input
        if self.p.sw_flag:
            self.m += self.p.dt*(
                    -self.p.gamma*f*(self.z[-1]) -
                    1/self.p.tau_w*(self.m - self.p.m_prior)
                    )
            self.w = self.m


class ClasLearnerS:
    """
    Classical synapse object (spiking feedback version). Uses online gradient
    descent on cross-entropy loss to update synaptic weights.

    Parameters
    ----------
    p : data class instance
        model and simulation parameters
    w : ndarray
        initial weights

    Attributes
    ----------
    p : data class instance
        model and simulation parameters
    m : ndarray
        dummy slow weights (mean)
    s2 : ndarray
        initial slow weights (variance; unused)
    dw : ndarray
         dummy fast weight variable (unused)
    w : ndarray
        weight variable
    h: ndarray
        convenience vector for use in plasticity rule
    z :  ndarray
        eligibility trace
    X : list
        buffer for maintaining lagged synaptic input
    input : ndarray
        padded synaptic input, scaled by time constant
    q0 : float
        spontaneous feedback rate

    """
    def __init__(self, p, w):
        self.p = p
        self.m = w
        self.w = w
        self.s2 = []
        self.dw = np.array(0.)

        self.z = np.zeros((3, p.num_syns))
        self.X = [np.zeros((p.M, p.num_syns)) for _ in range(int(p.lag/p.dt)+1)]
        self.input = np.zeros((3, p.num_syns))

        self.h = np.array([0, 0, 1]).reshape(-1, 1)
        self.q0 = self.p.eta0*1e-3*self.p.dt

    def update(self, x, f):
        """
        Integrates input and feedback, then updates synaptic weights.

        Parameters
        ----------
        x : ndarray
            synaptic input
        f : float
            error feedback

        """
        self.X.append(x)
        self.X.pop(0)
        self.input[0] = 1/self.p.tau_I*self.X[0]

        self.z += self.p.dt*self.p.A@self.z + self.input
        if self.p.sw_flag:
            self.m += (
                    -self.p.gamma*self.p.rho*(f - self.q0)*self.z[-1] -
                    self.p.dt/self.p.tau_w*(self.m - self.p.m_prior)
                    )
            self.w = self.m


def Ricatti_ss(p):
    """
    Solve steady-state matrix Ricatti equation for computing control gain.

    Parameters
    ----------
    p : data class instance
        model and simulation parameters

    Returns
    -------
    S : ndarray
        Ricatti equation solution
    L : ndarray
        control gain
    B : ndarray
        convenience matrix for computing control

    """
    Q = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]])
    B = np.array([[1/p.tau_I, 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    R = p.lam_u*np.eye(3)
    S = linalg.solve_continuous_are(p.A, B, Q, R)
    L = 1/p.lam_u*B@S
    return S, L, B


def Kalman_ss(p, lam):
    """
    Solve steady-state matrix Ricatti equation for computing Kalman gain

    Parameters
    ----------
    p : data class instance
        model and simulation parameters
    lam : ndarray
        noise covariance matrix

    Returns
    -------
    Sig2 : ndarray
        error estimate error-covariance matrix
    K : ndarray
        control gain

    """
    h = np.array([0, 0, 1]).reshape(-1, 1)
    Sig2 = linalg.solve_continuous_are(p.A.T, h, lam, p.sig2_f)
    K = 1/p.sig2_f*Sig2@h
    return Sig2, K


