
import numpy as np

def plot1d(mara, fields, show=True, **kwargs):
    import matplotlib.pyplot as plt
    lines = { }
    x, y, z = mara.coordinate_grid()
    try:
        axes = plot1d.axes
    except:
        plot1d.axes = [plt.subplot(len(fields),1,n+1) for n,f in enumerate(fields)]
        axes = plot1d.axes

    for ax, f in zip(axes, fields):
        lines[f], = ax.plot(x.flat, mara.fields[f], '-o', mfc='none', label=(
                f + ' ' + kwargs.get('label', '')))
    if show:
        for ax in axes:
            ax.legend(loc='best')
            yl = ax.get_ylim()
            ax.set_ylim(yl[0]+1e-8, yl[1]-1e-8)
        for ax in axes[:-1]:
            ax.set_xticks([])
        axes[-1].set_xlabel('position')
        plt.subplots_adjust(hspace=0.0, wspace=0)
        plt.show()
    return lines


def plot2d(mara, fields, show=True, **kwargs):
    import matplotlib.pyplot as plt
    lines = { }
    x, y, z = mara.coordinate_grid()
    ng = mara.number_guard_zones()
    try:
        axes = plot2d.axes
    except:
        nr = 2
        nc = np.ceil(len(fields) / float(nr))
        plot2d.axes = [plt.subplot(nc,nr,n+1) for n,f in enumerate(fields)]
        axes = plot2d.axes
    for ax, f in zip(axes, fields):
        #lines[f] = ax.imshow(mara.fields[f][ng:-ng,ng:-ng].T, interpolation='nearest')
        lines[f] = ax.imshow(mara.fields[f][:,:].T, interpolation='nearest')
    if show:
        for ax, f in zip(axes, fields):
            ax.set_title(f)
        plt.show()
    return lines


def plot3d(mara, fields, show=True, **kwargs):
    import matplotlib.pyplot as plt
    lines = { }
    x, y, z = mara.coordinate_grid()
    ng = mara.number_guard_zones()
    try:
        axes = plot2d.axes
    except:
        nr = 2
        nc = np.ceil(len(fields) / float(nr))
        plot2d.axes = [plt.subplot(nc,nr,n+1) for n,f in enumerate(fields)]
        axes = plot2d.axes
    for ax, f in zip(axes, fields):
        i0 = mara.fields[f].shape[0] / 2
        lines[f] = ax.imshow(mara.fields[f][i0,ng:-ng,ng:-ng].T, interpolation='nearest')
    if show:
        for ax, f in zip(axes, fields):
            ax.set_title(f)
        plt.show()
    return lines



if __name__ == "__main__":
    import optparse
    import numpy as np
    import h5py

    class Hdf5FileWrapper(h5py.File):
        def __init__(self, *args, **kwargs):
            super(Hdf5FileWrapper, self).__init__(*args, **kwargs)

        @property
        def fields(self):
            return self["prim"]

        @property
        def shape(self):
            return self["prim"][self["prim"].keys()[0]].shape

        def number_guard_zones(self):
            return 0

        def coordinate_grid(self):
            x0, y0, z0 = -0.5, -0.5, -0.5
            x1, y1, z1 = +0.5, +0.5, +0.5
            Nx, Ny, Nz = self.shape + (1,) * (3 - len(self.shape))
            dx, dy, dz = (x1 - x0) / Nx, (y1 - y0) / Ny, (z1 - z0) / Nz
            return np.mgrid[x0+dx/2 : x1+dx/2 : dx,
                            y0+dy/2 : y1+dy/2 : dy,
                            z0+dz/2 : z1+dz/2 : dz]

    parser = optparse.OptionParser(usage="plotting [opts] input.h5")
    opts, args = parser.parse_args()

    if not args:
        parser.print_usage()

    for arg in args:
        h5f = Hdf5FileWrapper(arg)
        plot1d(h5f, ["rho", "pre", "vx"], show=(arg is args[-1]))

