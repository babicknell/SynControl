"""
Core parameters for model and plasticity rule.
"""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Params:
    num_syns: int  # 1000
    M: int
    T: float  # (ms)
    dt: float  # (ms)
    tau_I: float
    tau_r: float
    tau_y: float
    tau_w: float
    alpha_Ir: float
    nu_bar: float
    m_prior: float  # (mV)
    s2_prior: float  # (mV^2)
    sig2_I: float
    sig2_r: float
    sig2_y: float
    sig2_f: float
    eta0: float
    rho: float
    gamma: float
    lam_u: float
    pop_control: str
    sig2_0: float = field(init=False)
    n_steps: int = field(init=False)
    A: np.ndarray = field(init=False)

    def __post_init__(self):
        self.sig2_0 = 2/self.tau_I*(1/self.tau_I*(self.num_syns - 1)*
                                    self.s2_prior*self.nu_bar + self.sig2_I)
        self.n_steps = int(self.T/self.dt)
        self.A = np.array([[-1./self.tau_I, 0., 0.],
                           [self.alpha_Ir/self.tau_r, -1./self.tau_r, 0.],
                           [0., 1./self.tau_y, -1./self.tau_y]])


def init_params():
    """ Create data structure of task parameters """
    params = Params(
        num_syns=1000,
        M=1,
        T=5e5,
        dt=0.1,
        tau_I=5.,
        tau_r=50.,
        tau_y=100.,
        tau_w=1e6,  # (ms)
        alpha_Ir=50.,
        nu_bar=1e-3*25.,
        m_prior=0.01,
        s2_prior=0.025,
        sig2_I=0.05,
        sig2_r=0.05,
        sig2_y=0.05,
        sig2_f=0.5,
        eta0=64.,
        rho=0.3,
        gamma=1e-4,
        lam_u=1.,
        pop_control='local'
    )
    return params


