

import numpy as np
from numpy.fft import *


class DrivingModule2d(object):
    theta = 1.0 # restoring parameter
    sigma = 1.0 # step size multiplier

    def __init__(self, shape, seed=12345):
        self._shape = tuple(shape)
        self._rng = np.random.RandomState(seed)
        self._sol = np.zeros(self._shape, dtype=np.complex)

    def advance(self, dt):
        dx = self._rng.normal(0.0, dt, size=self._shape)
        dy = self._rng.normal(0.0, dt, size=self._shape) * 1.j
        t, s = self.theta, self.sigma
        self._sol += -t * self._sol * dt + (s*dx + s*dy)

    def amplitude(self, k):
        k[abs(k) < 1e-12] = 1e-12
        k0 = 25
        return np.exp(-(k/k0 - 1.0)**2)

    def wave_number(self, magnitude=False):
        L = 1.0
        Nx, Ny = self._shape
        K = [fftfreq(Nx)[:,np.newaxis] * (2*np.pi*Nx/L),
             fftfreq(Ny)[np.newaxis,:] * (2*np.pi*Ny/L)]
        if magnitude:
            K.append((K[0]**2 + K[1]**2)**0.5)
        return K

    def rms(self):
        Fx, Fy = self.field
        return (Fx**2 + Fy**2).mean()**0.5

    def power_spectrum(self, bins=64):
        Kx, Ky, K = self.wave_number(magnitude=True)
        Fx, Fy = self.field
        Gx, Gy = fftn(Fx), fftn(Fy)
        w = abs(Gx)**2 + abs(Gy)**2
        bins = np.logspace(0, 3, bins)
        Nk, b = np.histogram(K, bins=bins)
        Pk, b = np.histogram(K, bins=bins, weights=w)
        Nk[Nk == 0] = 1.0
        return Pk/Nk, b

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
        L = 1.0
        Nx, Ny = self._shape
        Kx, Ky, K = self.wave_number(magnitude=True)
        fk = self.amplitude(K) / K * K.size
        fk[0,0] = 0.0
        Fx = ifftn(-1.j * Ky * self._sol * fk).real
        Fy = ifftn(+1.j * Kx * self._sol * fk).real
        return Fx, Fy


def test_power_spectrum(driving):
    import matplotlib.pyplot as plt
    Pk, bins = driving.power_spectrum(bins=64)
    x = 0.5*(bins[1:] + bins[:-1])
    plt.loglog(x, Pk, label='P')
    plt.show()


def test_streamlines(driving):
    import matplotlib.pyplot as plt
    from pyfish.streamplot import streamplot
    Fx, Fy = driving.field
    x = np.linspace(0.0, 1.0, Fx.shape[0])
    y = np.linspace(0.0, 1.0, Fy.shape[1])
    F = (Fx**2 + Fy**2)**0.5
    streamplot(x, y, Fx, Fy, density=1, color=F, linewidth=5*F/F.max())
    plt.show()


def test_image(driving):
    import matplotlib.pyplot as plt
    from pyfish.streamplot import streamplot
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
    from pyfish.streamplot import streamplot
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
    while t0 < 15.0:
        print t0
        driving.advance(dt)
        t0 += dt
        t.append(t0)
        A.append(driving._sol[N/2,N/2])
    plt.plot(t, [a.real for a in A])
    plt.plot(t, [a.imag for a in A])
    plt.show()


def test_power():
    for N in [4,8,16,32,64,128,256,512]:
        driving = DrivingModule2d([N,N])
        for i in range(24):
            driving.advance(0.1)
        print driving.rms()


if __name__ == "__main__":
    N = 256
    driving = DrivingModule2d([N,N])
    for i in range(24):
        print driving.rms()
        driving.advance(0.1)

    test_power_spectrum(driving)
    test_streamlines(driving)
    test_image(driving)
    test_divergence(driving)
    test_timevar()
