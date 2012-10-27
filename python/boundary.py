
import numpy as np


class BoundaryConditions(object):
    def set_boundary(self, X, ng, field='cons'):
        getattr(self, "set_boundary%dd" % (len(X.shape) - 1))(X, ng, field)

    def set_boundary1d(self, *args, **kwargs):
        raise NotImplementedError

    def set_boundary2d(self, *args, **kwargs):
        raise NotImplementedError

    def set_boundary3d(self, *args, **kwargs):
        raise NotImplementedError


class Outflow(BoundaryConditions):
    def __init__(self):
        pass

    def set_boundary1d(self, X, ng, field):
        X[:+ng] = X[+(ng+0)]
        X[-ng:] = X[-(ng+1)]

    def set_boundary2d(self, X, ng, field):
        X[:,:+ng] = X[:,+(ng+0)][:,np.newaxis,:]
        X[:,-ng:] = X[:,-(ng+1)][:,np.newaxis,:]
        X[:+ng,:] = X[+(ng+0),:][np.newaxis,:,:]
        X[-ng:,:] = X[-(ng+1),:][np.newaxis,:,:]

    def set_boundary3d(self, X, ng, field):
        X[:,:,:+ng] = X[:,:,+(ng+0)][:,:,np.newaxis,:]
        X[:,:,-ng:] = X[:,:,-(ng+1)][:,:,np.newaxis,:]
        X[:,:+ng,:] = X[:,+(ng+0),:][:,np.newaxis,:,:]
        X[:,-ng:,:] = X[:,-(ng+1),:][:,np.newaxis,:,:]
        X[:+ng,:,:] = X[+(ng+0),:,:][np.newaxis,:,:,:]
        X[-ng:,:,:] = X[-(ng+1),:,:][np.newaxis,:,:,:]


class Periodic(BoundaryConditions):
    def __init__(self):
        pass

    def set_boundary1d(self, X, ng, field):
        X[:+ng] = X[-2*ng:-ng]
        X[-ng:] = X[+ng:+2*ng]

    def set_boundary2d(self, X, ng, field):
        X[:+ng,:] = X[-2*ng:-ng,:]
        X[-ng:,:] = X[+ng:+2*ng,:]
        X[:,:+ng] = X[:,-2*ng:-ng]
        X[:,-ng:] = X[:,+ng:+2*ng]

    def set_boundary3d(self, X, ng, field):
        X[:+ng,:,:] = X[-2*ng:-ng,:,:]
        X[-ng:,:,:] = X[+ng:+2*ng,:,:]
        X[:,:+ng,:] = X[:,-2*ng:-ng,:]
        X[:,-ng:,:] = X[:,+ng:+2*ng,:]
        X[:,:,:+ng] = X[:,:,-2*ng:-ng]
        X[:,:,-ng:] = X[:,:,+ng:+2*ng]


"""
class Inflow(BoundaryConditions):
    def __init__(self, SL, SR):
        self.UL = np.array([S.conserved() for S in SL])
        self.UR = np.array([S.conserved() for S in SR])
        self.GL = np.array([S.gravity.copy() for S in SL])
        self.GR = np.array([S.gravity.copy() for S in SR])

    def set_boundary1d(self, mara):
        ng = mara.number_guard_zones()
        U = mara.fluid.conserved()
        G = mara.fluid.gravity
        U[:+ng] = self.UL
        U[-ng:] = self.UR
        G[:+ng] = self.GL
        G[-ng:] = self.GR
        mara.fluid.from_conserved(U)
"""
