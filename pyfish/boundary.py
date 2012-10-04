
import numpy as np


class BoundaryConditions(object):
    def set_boundary(self, mara):
        getattr(self, "set_boundary%dd" % len(mara.fluid.shape))(mara)

    def set_boundary1d(self, mara):
        raise NotImplementedError

    def set_boundary2d(self, mara):
        raise NotImplementedError

    def set_boundary3d(self, mara):
        raise NotImplementedError


class Outflow(BoundaryConditions):
    def __init__(self):
        pass

    def set_boundary1d(self, mara):
        ng = mara.number_guard_zones()
        U = mara.fluid.conserved()
        U[:+ng] = U[+(ng+0)]
        U[-ng:] = U[-(ng+1)]
        mara.fluid.from_conserved(U)

    def set_boundary2d(self, mara):
        ng = mara.number_guard_zones()
        U = mara.fluid.conserved()
        U[:,:+ng] = U[:,+(ng+0)][:,np.newaxis,:]
        U[:,-ng:] = U[:,-(ng+1)][:,np.newaxis,:]
        U[:+ng,:] = U[+(ng+0),:][np.newaxis,:,:]
        U[-ng:,:] = U[-(ng+1),:][np.newaxis,:,:]
        mara.fluid.from_conserved(U)

    def set_boundary3d(self, mara):
        ng = mara.number_guard_zones()
        U = mara.fluid.conserved()
        U[:,:,:+ng] = U[:,:,+(ng+0)][:,:,np.newaxis,:]
        U[:,:,-ng:] = U[:,:,-(ng+1)][:,:,np.newaxis,:]
        U[:,:+ng,:] = U[:,+(ng+0),:][:,np.newaxis,:,:]
        U[:,-ng:,:] = U[:,-(ng+1),:][:,np.newaxis,:,:]
        U[:+ng,:,:] = U[+(ng+0),:,:][np.newaxis,:,:,:]
        U[-ng:,:,:] = U[-(ng+1),:,:][np.newaxis,:,:,:]
        mara.fluid.from_conserved(U)


class Inflow(BoundaryConditions):
    def __init__(self, SL, SR):
        self.UL = np.array([S.conserved() for S in SL])
        self.UR = np.array([S.conserved() for S in SR])
        self.GL = np.array([S.gravity.copy() for S in SL])
        self.GR = np.array([S.gravity.copy() for S in SR])

    def set_boundary1d(self, mara):
        ng = mara.number_guard_zones()
        U = mara.fluid.conserved()
        U[:+ng] = self.UL
        U[-ng:] = self.UR
        G[:+ng] = self.GL
        G[-ng:] = self.GR
        mara.fluid.from_conserved(U)


class Periodic(BoundaryConditions):
    def __init__(self):
        pass

    def set_boundary1d(self, mara):
        ng = mara.number_guard_zones()
        U = mara.fluid.conserved()
        G = mara.fluid.gravity
        for X in [U, G]:
            X[:+ng] = X[-2*ng:-ng]
            X[-ng:] = X[+ng:+2*ng]
        mara.fluid.from_conserved(U)
