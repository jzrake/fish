
import numpy as np

class Outflow(object):
    def __init__(self):
        pass

    def set_boundary(self, mara, U):
        getattr(self, "set_boundary%dd" % (len(U.shape) - 1))(mara, U)

    def set_boundary1d(self, mara, U):
        ng = mara.number_guard_zones()
        U[:+ng] = U[+(ng+0)]
        U[-ng:] = U[-(ng+1)]

    def set_boundary2d(self, mara, U):
        ng = mara.number_guard_zones()
        U[:,:+ng] = U[:,+(ng+0)][:,np.newaxis,:]
        U[:,-ng:] = U[:,-(ng+1)][:,np.newaxis,:]
        U[:+ng,:] = U[+(ng+0),:][np.newaxis,:,:]
        U[-ng:,:] = U[-(ng+1),:][np.newaxis,:,:]

    def set_boundary3d(self, mara, U):
        ng = mara.number_guard_zones()
        U[:,:,:+ng] = U[:,:,+(ng+0)][:,:,np.newaxis,:]
        U[:,:,-ng:] = U[:,:,-(ng+1)][:,:,np.newaxis,:]
        U[:,:+ng,:] = U[:,+(ng+0),:][:,np.newaxis,:,:]
        U[:,-ng:,:] = U[:,-(ng+1),:][:,np.newaxis,:,:]
        U[:+ng,:,:] = U[+(ng+0),:,:][np.newaxis,:,:,:]
        U[-ng:,:,:] = U[-(ng+1),:,:][np.newaxis,:,:,:]


class Inflow(object):
    def __init__(self, SL, SR):
        self.UL = np.array([S.conserved() for S in SL])
        self.UR = np.array([S.conserved() for S in SR])

    def set_boundary(self, mara, U):
        getattr(self, "set_boundary%dd" % (len(U.shape) - 1))(mara, U)

    def set_boundary1d(self, mara, U):
        ng = mara.number_guard_zones()
        U[:+ng] = self.UL
        U[-ng:] = self.UR

    def set_boundary2d(self, mara, U):
        raise NotImplementedError

    def set_boundary3d(self, mara, U):
        raise NotImplementedError
