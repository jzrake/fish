
import numpy as np
from Mara.capis import FluidDescriptor
from Mara import boundary, driving, gravity
from Mara.utils import init_options


class TestProblem(object):
    __metaclass__ = init_options
    tfinal = 1.0
    lower_bounds = [-0.5, -0.5, -0.5]
    upper_bounds = [+0.5, +0.5, +0.5]
    resolution = [128]
    plot_fields = ['rho', 'pre', 'vx']
    parallel = False
    CFL = 0.3
    cpi = 1.0 # checkpoint interval

    def __init__(self, **kwargs):
        for k,v in kwargs.iteritems():
            setattr(self, k, v)
        self._descr = FluidDescriptor(fluid=self.fluid, gamma=self.gamma)

    @property
    def fluid_descriptor(self):
        return self._descr

    def build_boundary(self, mara):
        return boundary.Outflow()

    def ginit(self, x, y, z):
        return 0.0

    def keep_running(self, status):
        return status.time_current < self.tfinal

    def __setattr__(self, k, v):
        """
        Allow setting list attributes with a comma-separated string
        """
        if hasattr(self, k):
            if type(getattr(self, k)) is list:
                tp = type(getattr(self, k)[0])
                if type(v) is str:
                    v = [tp(vi) for vi in v.split(',')]
        super(TestProblem, self).__setattr__(k, v)


class TwoStateProblem(TestProblem):
    __metaclass__ = init_options
    fluid = 'nrhyd'
    gamma = 1.4
    tfinal = 0.2
    geometry = 'planar'
    direction = 'x'

    def __init__(self, *args, **kwargs):
        super(TwoStateProblem, self).__init__(*args, **kwargs)
        if len(self.resolution) == 2:
            self.plot_fields.append('vy')

    def pinit(self, x, y, z):
        if self.geometry == 'planar':
            r = {'x': x, 'y': y, 'z': z}[self.direction]
            R = 0.0
        elif self.geometry == 'cylindrical':
            r = (x**2 + y**2)**0.5
            R = 0.125
        elif self.geometry == 'spherical':
            r = (x**2 + y**2 + z**2)**0.5
            R = 0.125
        else:
            raise ValueError("invalid problem geometry: %s" % self.geometry)
        if r > R:
            return self.state2
        else:
            return self.state1


class Shocktube1(TwoStateProblem):
    state1 = [1.000, 1.000, 0.000, 0.0, 0.0]
    state2 = [0.125, 0.100, 0.000, 0.0, 0.0]


class Shocktube2(TwoStateProblem):
    state1 = [1.000, 0.400,-2.000, 0.0, 0.0]
    state2 = [1.000, 0.400, 2.000, 0.0, 0.0]


class Shocktube3(TwoStateProblem):
    state1 = [1.0, 1e+3, 0.0, 0.0, 0.0]
    state2 = [1.0, 1e-2, 0.0, 0.0, 0.0]


class Shocktube4(TwoStateProblem):
    state1 = [1.0, 1e-2, 0.0, 0.0, 0.0]
    state2 = [1.0, 1e+2, 0.0, 0.0, 0.0]


class Shocktube5(TwoStateProblem):
    state1 = [5.99924, 460.894, 19.59750, 0.0, 0.0]
    state2 = [5.99924,  46.095, -6.19633, 0.0, 0.0]


class ContactWave(TwoStateProblem):
    state1 = [1.0, 1.0, 0.0, 0.7, 0.2]
    state2 = [0.1, 1.0, 0.0, 0.7, 0.2]


class SrhdCase1_DFIM98(TwoStateProblem):
    fluid = 'srhyd'
    state1 = [10.0, 13.30, 0.0, 0.0, 0.0]
    state2 = [ 1.0,  1e-6, 0.0, 0.0, 0.0]


class SrhdCase2_DFIM98(TwoStateProblem):
    fluid = 'srhyd'
    state1 = [1, 1e+3, 0.0, 0.0, 0.0]
    state2 = [1, 1e-2, 0.0, 0.0, 0.0]


class SrhdHardTransverse_RAM(TwoStateProblem):
    fluid = 'srhyd'
    state1 = [1, 1e+3, 0.0, 0.9, 0.0]
    state2 = [1, 1e-2, 0.0, 0.9, 0.0]


class OneDimensionalUpsidedownGaussian(TestProblem):
    '''
    A good test for test-fluid gravity. Should only work when gravity is
    implemented as a momentum and energy source term, i.e. when $del^2 phi =
    rho$ does not have to be satisfied. Sets up an upside-down Gaussian
    potential well with a pressure profile in hydrostatic equillibrium.
    '''
    __metaclass__ = init_options
    fluid = 'gravs'
    gamma = 1.4
    sig = 0.05
    sie = 2.00
    ph0 = 1.0

    def pinit(self, x, y, z):
        phi = self.ginit(x, y, z)[0]
        e0 = self.sie
        D0 = 1.0
        rho = D0 * np.exp(-phi / (e0 * (self.gamma - 1.0)))
        pre = rho * e0 * (self.gamma - 1.0)
        return [rho, pre, 0.0, 0.0, 0.0]

    def ginit(self, x, y, z):
        phi = -self.ph0 * np.exp(-0.5 * x**2 / self.sig**2)
        gph = -x/self.sig**2 * phi
        return [phi, gph, 0.0, 0.0]

    def build_boundary(self, mara):
        ng = mara.number_guard_zones()
        return boundary.Inflow(mara.fluid[0:ng], mara.fluid[-ng:])


class OneDimensionalPolytrope(TestProblem):
    '''
    Provides the initial conditions of a 1d polytrope for Gamma=2. Pressure
    density satisfy Poisson equation and hydrostatic equillibrium.
    '''
    __metaclass__ = init_options
    fluid = 'gravs' # or gravp, grave
    gamma = 2.0
    Dc = 1.0 # central density
    Da = 1e-2 # atmosphere density
    Pa = 1e-3 # atmosphere pressure
    R = 0.25
    four_pi_G = 1.0
    plot_fields = ['rho', 'pre', 'phi', 'gph']
    pauls_fix = False
    gaussian = False

    def __init__(self, *args, **kwargs):
        super(OneDimensionalPolytrope, self).__init__(*args, **kwargs)
        if self.selfgrav:
            self.poisson_solver = gravity.PoissonSolver1d()

    def pinit(self, x, y, z):
        R = self.R
        K = 0.5 * R**2 / np.pi**2
        if self.gaussian:
            rho = self.Dc * np.exp(-x*x/0.01) + self.Da
            pre = K * rho**1.0
        elif abs(x) < R/2:
            rho = self.Dc * np.cos(np.pi * x / R)
            pre = K * rho**2.0
        else:
            if self.pauls_fix:
                rho = 0.0
                pre = 0.0
            else:
                rho = self.Da
                pre = K * rho**2.0
        if self.pauls_fix:
            if rho < self.Da: rho = self.Da
            if pre < self.Pa: pre = self.Pa
        return [rho, pre, 0.0, 0.0, 0.0]

    def ginit(self, x, y, z):
        return [0.0, 0.0, 0.0, 0.0]

    def build_boundary(self, mara):
        return boundary.Periodic()


class PeriodicDensityWave(TestProblem):
    '''
    Gives a stationary density wave without pressure variation.
    '''
    __metaclass__ = init_options
    fluid = 'nrhyd'
    gamma = 1.4
    p0 = 1.00 # background pressure
    D0 = 10.00 # background density
    D1 = 1.00 # density fluctuation
    v0 = 0.00 # velocity of the wave
    n0 = 4 # integer valued wave-number
    khat = [1, 0, 0] # unit vector of plane-wave direction (integers please)
    plot_fields = ['rho', 'vx']

    def __init__(self, *args, **kwargs):
        super(PeriodicDensityWave, self).__init__(*args, **kwargs)
        if self.fluid in ['gravs', 'gravp']:
            self.plot_fields.append('phi')
            self.poisson_solver = gravity.PoissonSolver1d()
            self.fluid_descriptor.rhobar = self.D0
        k = np.dot(self.khat, self.khat)**0.5

    def pinit(self, x, y, z):
        L = self.upper_bounds[0] - self.lower_bounds[0]
        r = np.dot(self.khat, [x, y, z])
        rho = self.D0 + self.D1 * np.cos(2 * self.n0 * np.pi * r / L)
        v = [self.v0 * k for k in self.khat]
        return [rho, self.p0] + v

    def ginit(self, x, y, z):
        return [0.0, 0.0, 0.0, 0.0]

    def build_boundary(self, mara):
        return boundary.Periodic()


class DrivenTurbulence1d(TestProblem):
    __metaclass__ = init_options
    fluid = 'nrhyd'
    gamma = 1.4
    tfinal = 1.0
    resolution = [128]
    plot_fields = ['rho', 'pre', 'vx']
    def __init__(self, *args, **kwargs):
        super(DrivenTurbulence1d, self).__init__(*args, **kwargs)
        self.driving = driving.DrivingModule1d()
        if len(self.resolution) != 1:
            raise ValueError("problem needs a 1d domain")

    def pinit(self, x, y, z):
        return [1.0, 1.0, 0.0, 0.0, 0.0]

    def build_boundary(self, mara):
        return boundary.Periodic()


class DrivenTurbulence2d(TestProblem):
    __metaclass__ = init_options
    fluid = 'nrhyd'
    gamma = 1.4
    tfinal = 1.0
    resolution = [128, 128]
    plot_fields = ['rho', 'pre', 'vx', 'vy']
    def __init__(self, *args, **kwargs):
        super(DrivenTurbulence2d, self).__init__(*args, **kwargs)
        self.driving = driving.DrivingModule2d()
        if len(self.resolution) != 2:
            raise ValueError("problem needs a 2d domain")

    def pinit(self, x, y, z):
        return [1.0, 1.0, 0.0, 0.0, 0.0]

    def build_boundary(self, mara):
        return boundary.Periodic()


class DrivenTurbulence3d(TestProblem):
    __metaclass__ = init_options
    fluid = 'nrhyd'
    gamma = 1.4
    tfinal = 1.0
    resolution = [16, 16, 16]
    def __init__(self, *args, **kwargs):
        super(DrivenTurbulence3d, self).__init__(*args, **kwargs)
        self.driving = driving.DrivingModule3d()
        if len(self.resolution) != 3:
            raise ValueError("problem needs a 3d domain")

    def pinit(self, x, y, z):
        return [1.0, 1.0, 0.0, 0.0, 0.0]

    def build_boundary(self, mara):
        return boundary.Periodic()


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


def ginit(self, x, y, z):
    """
    An attempt at a static gravity solution to the 1d polytrop. In practive we
    just use the FFT.
    """
    R = self.R
    s = (2 * self.Dc * R) / np.pi # enclosed mass per unit area
    if abs(x) < R/2:
        phi = -(self.Dc / (np.pi/R)**2) * np.cos(np.pi*x/R) * self.four_pi_G
        gph = +(self.Dc / (np.pi/R)**1) * np.sin(np.pi*x/R) * self.four_pi_G
    else:
        phi = abs(x) * self.four_pi_G * s
        gph = self.four_pi_G * s * np.sign(x)
    return [phi, gph, 0.0, 0.0]


def get_problem_class():
    import problems
    opts = { }
    for k,v in problems.__dict__.iteritems():
        if type(v) is type and v not in [problems.TestProblem,
                                         problems.TwoStateProblem]:
            if issubclass(v, problems.TestProblem):
                opts[k] = v
    for n, k in enumerate(sorted(opts.keys())):
        print "[%2d]: %s" % (n, k)

    def ask():
        try:
            return opts[sorted(opts.keys())[input("enter a problem: ")]]
        except IndexError:
            return ask()
    return ask()
