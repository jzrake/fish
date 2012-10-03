
import numpy as np
import pyfluids
from pyfish import boundary


class TestProblem(object):
    tfinal = 1.0
    def __init__(self, **kwargs):
        for k,v in kwargs.iteritems():
            setattr(self, k, v)
        self._descr = pyfluids.FluidDescriptor(fluid=self.fluid,
                                               gamma=self.gamma)

    @property
    def fluid_descriptor(self):
        return self._descr

    def build_boundary(self, mara):
        return boundary.Outflow()

    def ginit(self, x, y, z):
        return 0.0


class OneDimensionalUpsidedownGaussian(TestProblem):
    '''
    A good test for test-fluid gravity. Should only work when gravity is
    implemented as a momentum and energy source term, i.e. when $del^2 phi =
    rho$ does not have to be satisfied. Sets up an upside-down Gaussian
    potential well with a pressure profile in hydrostatic equillibrium.
    '''
    fluid = 'gravs'
    gamma = 1.4
    sig = 0.05
    sie = 2.00
    ph0 = 1.0
    def ginit(self, x, y, z):
        phi = -self.ph0 * np.exp(-0.5 * x**2 / self.sig**2)
        gph = -x/self.sig**2 * phi
        return [phi, gph, 0.0, 0.0]

    def pinit(self, x, y, z):
        phi = self.ginit(x, y, z)[0]
        e0 = self.sie
        D0 = 1.0
        rho = D0 * np.exp(-phi / (e0 * (self.gamma - 1.0)))
        pre = rho * e0 * (self.gamma - 1.0)
        return [rho, pre, 0.0, 0.0, 0.0]

    def build_boundary(self, mara):
        ng = mara.number_guard_zones()
        return boundary.Inflow(mara.fluid[0:ng], mara.fluid[-ng:])


class OneDimensionalPolytrope(TestProblem):
    '''
    Provides the initial conditions of a 1d polytrope for Gamma=2. Pressure
    density satisfy Poisson equation and hydrostatic equillibrium.
    '''
    fluid = 'gravs' # or gravp, grave
    gamma = 2.0
    D0 = 1.0
    R = 1.1
    def ginit(self, x, y, z):
        R = self.R
        phi = -(self.D0 / (np.pi/R)**2) * np.cos(np.pi * x / R)
        gph = +(self.D0 / (np.pi/R)**1) * np.sin(np.pi * x / R)
        return [phi, gph, 0.0, 0.0]

    def pinit(self, x, y, z):
        R = self.R
        K = 0.5 * R**2 / np.pi**2
        rho = self.D0 * np.cos(np.pi * x / R)
        if rho < 1e-8:
            rho = 1e-8 # prevent zero density
        pre = K * rho**2.0
        return [rho, pre, 0.0, 0.0, 0.0]

    def build_boundary(self, mara):
        ng = mara.number_guard_zones()
        return boundary.Inflow(mara.fluid[0:ng], mara.fluid[-ng:])


class BrioWuShocktube(TestProblem):
    fluid = 'nrhyd'
    gamma = 1.4
    tfinal = 0.2
    def pinit(self, x, y, z):
        if x > 0.0:
            return [0.125, 0.100, 0.0, 0.0, 0.0]
        else:
            return [1.000, 1.000, 0.0, 0.0, 0.0]





def polytrope3d(x, y, z):
    rho_c = 1.0    # central density
    rho_f = 1.0e-3 # floor (atmospheric) density
    G = 1.0        # gravitational constant
    b = 0.3        # beta, stellar radius
    a = b / np.pi  # alpha
    n = 1.0        # polytropic index
    K = 4*np.pi*G * a**2 / ((n + 1) * rho_c**(1.0/n - 1.0))
    r = (x**2 + y**2 + z**2)**0.5 / a
    if r < 1e-6:
        rho = rho_c
    elif r >= np.pi:
        rho = rho_f
    else:
        rho = rho_c * np.sin(r) / r
    pre = K * rho**2
    return [rho, pre, 0.0, 0.0, 0.0]


def central_mass3d(x, y, z):
    rho_c = 1.0    # central density
    rho_f = 1.0e-2 # floor (atmospheric) density
    a = 0.3        # alpha, stellar radius
    r = (x**2 + y**2 + z**2)**0.5 / a
    if r < 0.5:
        rho = rho_c
    else:
        rho = rho_f
    pre = 1.0
    return [rho, pre, 0.0, 0.0, 0.0]
