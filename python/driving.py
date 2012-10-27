
import numpy as np
from numpy.fft import *


class DrivingModule3d(object):
    """
    Utilizes the gaussfield C++ Python extension to create serializable Gauss
    random fields:

    https://github.com/jzrake/gaussfield
    """
    def __init__(self):
        import gaussfield
        self._field = gaussfield.GaussianField3d(1.0, 1.0, 3, 12345)
        self._calls = 0

    def advance(self, dt):
        self._field.advance(dt)

    def sample(self, X, Y, Z):
        return self._field.sample(X, Y, Z)

    def drive(self, mara, dt):
        """
        Modifies the primitives according to the acceleration field F.
        """
        self._calls += 1
        if self._calls % 10 == 0 or not hasattr(self, '_F'):
            X, Y, Z = mara.coordinate_grid()
            start = time.clock()
            self._F = self.sample(X, Y, Z)
        mara.fluid.primitive[...,2:5] += self._F * dt


class DrivingModule2d(object):
    """
    The power spectrum in this module refers to the 2d profile dP/dk^2, not the 1d
    power spectrum dP/dk.
    """
    theta = 1.0 # restoring parameter
    sigma = 1.0 # step size multiplier
    L = 1.0 # domain size

    def __init__(self, shape, rms=1.0, seed=12345):
        self._shape = tuple(shape)
        self._rng = np.random.RandomState(seed)
        self._sol = np.zeros(self._shape, dtype=np.complex)
        self._rms = rms
        Kx, Ky, K = self.wave_number()
        self._totpower = self.power_spectrum(K, normalized=False).mean()

    def advance(self, dt):
        dx = self._rng.normal(0.0, (2*dt)**0.5, size=self._shape)
        dy = self._rng.normal(0.0, (2*dt)**0.5, size=self._shape) * 1.j
        t, s = self.theta, self.sigma
        self._sol += -t * self._sol * dt + (s*dx + s*dy)

    def wave_number(self):
        L = self.L
        Nx, Ny = self._shape
        K = [fftfreq(Nx)[:,np.newaxis],
             fftfreq(Ny)[np.newaxis,:]]
        K.append((K[0]**2 + K[1]**2)**0.5)
        return K

    def total_power(self, actual=False):
        if actual:
            Fx, Fy = self.field
            return (Fx**2 + Fy**2).mean()
        else:
            Kx, Ky, K = self.wave_number()
            return self.power_spectrum(K).mean()

    def power_spectrum(self, k=None, normalized=True, bins=None):
        if bins is None:
            k[abs(k) < 1e-12] = 1e-12
            k0 = 0.1
            Pk = np.exp(-(k/k0)**2) * (k/k0)**8
            if normalized:
                return Pk / self._totpower * self._rms**2
            else:
                return Pk
        else:
            Kx, Ky, K = self.wave_number()
            Fx, Fy = self.field
            Gx, Gy = fftn(Fx), fftn(Fy)
            w = abs(Gx)**2 + abs(Gy)**2
            w /= w.size
            K[0,0] = 1.0
            bins = np.logspace(np.log10(K.min()), np.log10(K.max()), bins)
            Nk, b = np.histogram(K, bins=bins)
            Pk, b = np.histogram(K, bins=bins, weights=w)
            Nk[Nk == 0] = 1.0
            return Pk / Nk, b

    def source_terms(self, P):
        Fx, Fy = self.field
        S = np.zeros_like(P)
        S[...,0] = 0.0
        S[...,1] = P[...,0] * (Fx*P[...,2] + Fy*P[...,3])
        S[...,2] = P[...,0] * Fx
        S[...,3] = P[...,0] * Fy
        S[...,4] = 0.0
        return S

    @property
    def field(self):
        Nx, Ny = self._shape
        Kx, Ky, K = self.wave_number()
        P = self.power_spectrum(K)
        fk = (P / K**2 * K.size)**0.5
        fk[0,0] = 0.0
        Fx = ifftn(-1.j * Ky * self._sol * fk).real
        Fy = ifftn(+1.j * Kx * self._sol * fk).real
        return Fx, Fy


def test_power_spectrum(driving):
    import matplotlib.pyplot as plt
    Pk, bins = driving.power_spectrum(bins=32)
    k = 0.5*(bins[1:] + bins[:-1])
    plt.loglog(k, Pk, label='P')
    plt.loglog(k, driving.power_spectrum(k), label='amplitude')
    plt.show()


def test_pdf(driving):
    import matplotlib.pyplot as plt
    from scipy.optimize import leastsq
    Fx, Fy = driving.field
    kwargs = dict(bins=200, histtype='stepfilled', normed=True, alpha=0.5)
    Nx, binsx, patchesx = plt.hist(Fx.flat, **kwargs)
    Ny, binsy, patchesy = plt.hist(Fy.flat, **kwargs)

    fitfunc = lambda p, x: (2*np.pi*p[1]**2)**-0.5 * np.exp(-(x - p[0])**2 / (2*p[1]**2))
    errfunc = lambda p, x, y: fitfunc(p, x) - y

    x = 0.5*(binsx[1:] + binsx[:-1])
    p0 = [0.0, 1.0]
    p1, success = leastsq(errfunc, p0, args=(x, Nx))
    print "mu = %f, sigma = %f" % tuple(p1)
    plt.plot(x, [fitfunc(p1, xi) for xi in x])
    plt.show()


def test_streamlines(driving):
    import matplotlib.pyplot as plt
    from streamplot import streamplot
    Fx, Fy = driving.field
    x = np.linspace(0.0, 1.0, Fx.shape[0])
    y = np.linspace(0.0, 1.0, Fy.shape[1])
    F = (Fx**2 + Fy**2)**0.5
    streamplot(x, y, Fx, Fy, density=1, color=F, linewidth=5*F/F.max())
    plt.show()


def test_image(driving):
    import matplotlib.pyplot as plt
    ax1 = plt.figure(1).add_subplot(111)
    ax2 = plt.figure(2).add_subplot(111)
    ax3 = plt.figure(3).add_subplot(111)
    Fx, Fy = driving.field
    F = (Fx**2 + Fy**2)**0.5
    ax1.imshow(F)
    ax2.imshow(Fx)
    ax3.imshow(Fy)
    plt.show()


def test_divergence(driving):
    import matplotlib.pyplot as plt
    from streamplot import streamplot
    Fx, Fy = driving.field
    Gx, Gy = fftn(Fx), fftn(Fy)
    Kx, Ky = driving.wave_number()
    F = (Fx**2 + Fy**2)**0.5
    divF = (np.gradient(Fx)[0] + np.gradient(Fy)[1])[1:-1,1:-1]
    crlF = (np.gradient(Fx)[1] - np.gradient(Fy)[0])[1:-1,1:-1]
    d = divF
    plt.imshow(d / F.mean())
    plt.colorbar()
    plt.show()


def test_timevar():
    import matplotlib.pyplot as plt
    N = 128
    driving = DrivingModule2d([N,N])
    A = [ ]
    t = [ ]
    dt = 1e-2
    t0 = 0.0
    while t0 < 25.0:
        print t0
        driving.advance(dt)
        t0 += dt
        t.append(t0)
        A.append(driving._sol[N/4,N/4])
    print "mean amplitude:", np.mean([abs(a) for a in A])
    plt.plot(t, [a.real for a in A])
    plt.plot(t, [a.imag for a in A])
    plt.show()


def test_power():
    for N in [4,8,16,32,64,128,256,512]:
        driving = DrivingModule2d([N,N])
        for i in range(48):
            driving.advance(0.1)
        print driving.total_power()


if __name__ == "__main__":
    N = 256
    driving = DrivingModule2d([N,N], rms=1.0)
    for i in range(12):
        print driving.total_power(), driving.total_power(actual=True)
        driving.advance(0.1)

    #test_power_spectrum(driving)
    test_pdf(driving)
    #test_streamlines(driving)
    #test_image(driving)
    #test_divergence(driving)
    #test_timevar()
    #test_power()
