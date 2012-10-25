
from pyfish import FishSolver
from pyfish import problems
from pyfish.simulation import MaraEvolutionOperator
from pyfish.plotting import *


class SimulationStatus:
    pass


def main():
    # Problem options
    problem_cfg = dict(resolution=[128],
                       tfinal=2.8, v0=0.0, gamma=1.001,
                       fluid='gravp', pauls_fix=False, gaussian=True)
    problem = problems.OneDimensionalPolytrope(selfgrav=True, **problem_cfg)
    #problem = problems.BrioWuShocktube(fluid='nrhyd',
    #                                   tfinal=0.2,
    #                                   geometry='planar', direction='x',
    #                                   resolution=[128])
    #problem = problems.PeriodicDensityWave(**problem_cfg)
    #problem = problems.DrivenTurbulence2d(tfinal=0.01)

    # Status setup
    status = SimulationStatus()
    status.CFL = 0.6
    status.iteration = 0
    status.time_step = 0.0
    status.time_current = 0.0
    status.chkpt_number = 0
    status.chkpt_last = 0.0
    status.chkpt_interval = 2.0
    measlog = { }

    # Scheme setup
    scheme = FishSolver()
    scheme.solver_type = ["godunov", "spectral"][0]
    scheme.reconstruction = "plm"
    scheme.riemann_solver = "hll"
    scheme.shenzha10_param = 100.0
    scheme.smoothness_indicator = ["jiangshu96", "borges08", "shenzha10"][2]

    # Plotting options
    plot_fields = problem.plot_fields
    plot_interactive = False
    plot_initial = True
    plot_final = True
    plot = [plot1d, plot2d, plot3d][len(problem.resolution) - 1]

    # Runtime options
    parallel = False

    if parallel:
        from pyfish.parallel import ParallelSimulation
        mara = ParallelSimulation(problem, scheme)
    else:
        mara = MaraEvolutionOperator(problem, scheme)

    mara.initial_model(problem.pinit, problem.ginit)

    if plot_interactive:
        import matplotlib.pyplot as plt
        plt.ion()
        lines = plot(mara, plot_fields, show=False)

    if plot_initial and not parallel:
        plot(mara, plot_fields, show=False, label='start')

    while status.time_current < problem.tfinal:
        if plot_interactive and not parallel:
            for f in plot_fields:
                lines[f].set_ydata(mara.fields[f])
            plt.draw()

        dt = mara.timestep(status.CFL)
        try:
            wall_step = mara.advance(dt, rk=3)
        except RuntimeError as e:
            print e
            break

        status.time_step = dt
        status.time_current += status.time_step
        status.iteration += 1

        status.message = "%05d(%d): t=%5.4f dt=%5.4e %5.1fkz/s %3.2fus/(z*Nq)" % (
            status.iteration, 0, status.time_current, dt,
            (mara.fluid.size / wall_step) * 1e-3,
            (wall_step / (mara.fluid.size*5)) * 1e6)

        if status.time_current - status.chkpt_last > status.chkpt_interval:
            status.chkpt_last = status.time_current
            status.chkpt_number += 1
            mara.write_checkpoint(status, dir="data/test", measlog=measlog)

        measlog[status.iteration] = mara.measure()
        measlog[status.iteration]["time"] = status.time_current
        measlog[status.iteration]["message"] = status.message
        print status.message

    mara.set_boundary()
    if plot_final and not parallel:
        plot(mara, plot_fields, show=True, label='end')



if __name__ == "__main__":
    main()
