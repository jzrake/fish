
import optparse
import imp
import sys
import time
from Mara.capis import FishSolver
from Mara.simulation import MaraEvolutionOperator
from Mara.plotting import *
from Mara import problems


class SimulationStatus: pass

class PlottingOptions:
    interactive = False
    initial     = False
    final       = False
    fields      = None


def main(problem, scheme, plot):
    # Status setup
    status = SimulationStatus()
    status.CFL = 0.8
    status.iteration = 0
    status.time_step = 0.0
    status.time_current = 0.0
    status.chkpt_number = 0
    status.chkpt_last = 0.0
    status.chkpt_interval = 0.1
    status.clock_start = time.clock()
    status.accum_wall = 0.0
    measlog = { }

    # Plotting options
    if not plot.fields:
        plot.fields = problem.plot_fields
    plot.func = [plot1d, plot2d, plot3d][len(problem.resolution) - 1]

    if problem.parallel:
        from Mara.parallel import ParallelSimulation
        mara = ParallelSimulation(problem, scheme)
    else:
        mara = MaraEvolutionOperator(problem, scheme)
    mara.initial_model(problem.pinit, problem.ginit)

    if plot.interactive:
        import matplotlib.pyplot as plt
        plt.ion()
        lines = plot.func(mara, plot.fields, show=False)

    if plot.initial and not problem.parallel:
        plot.func(mara, plot.fields, show=False, label='start')

    while problem.keep_running(status):
        if plot.interactive and not problem.parallel:
            for f in plot.fields:
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
        status.accum_wall += wall_step

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

    print "\n"
    print "performance report:"
    print "-------------------"
    print "wall time in integrations : %3.2f s" % status.accum_wall
    print "wall time total           : %3.2f s" % (time.clock() - status.clock_start)
    print "mean kz/s per iteration   : %3.2f kz/s" % (
        1e-3 * mara.fluid.size / (status.accum_wall+1e-14) * status.iteration)
    print "\n"

    mara.set_boundary()
    if plot.final and not problem.parallel:
        plot.func(mara, plot.fields, show=True, label='end')



if __name__ == "__main__":
    parser = optparse.OptionParser()
    opts, args = parser.parse_args()

    problem_cfg = { }
    scheme = FishSolver()
    plot = PlottingOptions()

    try:
        runparam = imp.load_source("runparam", args[0])

        for k,v in getattr(runparam, 'problem_cfg', { }).items():
            problem_cfg[k] = v

        for k,v in getattr(runparam, 'scheme_cfg', { }).items():
            setattr(scheme, k, v)

        for k,v in getattr(runparam, 'plotting_cfg', { }).items():
            setattr(plot, k, v)

    except IndexError: # no args[0]
        pass

    problem_class = problems.get_problem_class()
    problem = problem_class(**problem_cfg)
    main(problem, scheme, plot)
