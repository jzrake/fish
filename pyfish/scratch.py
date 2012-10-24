

class PythonDriveSweeps(object):
    """
    A Python implementation of the time derivative function in fish.c
    """
    def _dUdt1d(self, fluid, scheme):
        Nx, = self.fluid.shape
        dx, = self.dx,
        L = np.zeros_like(fluid.primitive)
        Fiph = scheme.intercell_flux(fluid[:], dim=0)
        L[1:] += -(Fiph[1:] - Fiph[:-1]) / dx
        return L

    def _dUdt2d(self, fluid, scheme):
        Nx, Ny = self.fluid.shape
        dx, dy = self.dx, self.dy
        L = np.zeros_like(fluid.primitive)
        for j in range(Ny):
            Fiph = scheme.intercell_flux(fluid[:,j], dim=0)
            L[1:,j] += -(Fiph[1:] - Fiph[:-1]) / dx
        for i in range(Nx):
            Giph = scheme.intercell_flux(fluid[i,:], dim=1)
            L[i,1:] += -(Giph[1:] - Giph[:-1]) / dy
        return L

    def _dUdt3d(self, fluid, scheme):
        Nx, Ny, Nz = self.fluid.shape
        dx, dy, dz = self.dx, self.dy, self.dz
        L = np.zeros_like(fluid.primitive)
        for j in range(Ny):
            for k in range(Nz):
                Fiph = scheme.intercell_flux(fluid[:,j,k], dim=0)
                L[1:,j,k] += -(Fiph[1:] - Fiph[:-1]) / dx
        for k in range(Nz):
            for i in range(Nx):
                Giph = scheme.intercell_flux(fluid[i,:,k], dim=1)
                L[i,1:,k] += -(Giph[1:] - Giph[:-1]) / dy
        for i in range(Nx):
            for j in range(Ny):
                Hiph = scheme.intercell_flux(fluid[i,j,:], dim=2)
                L[i,j,1:] += -(Hiph[1:] - Hiph[:-1]) / dz
        return L
