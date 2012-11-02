#!/usr/bin/env python

import optparse
import imp
import sys
import time
from Mara.capis import FishSolver
from Mara.simulation import MaraEvolutionOperator
from Mara.plotting import *
from Mara.utils import init_options
from Mara import problems


class SimulationStatus: pass

class PlottingOptions:
    __metaclass__ = init_options
    interactive = False
    initial     = False
    final       = False
    fields      = None


def main(problem, scheme, plot):
    # Status setup
    status = SimulationStatus()
    status.CFL = problem.CFL
    status.iteration = 0
    status.time_step = 0.0
    status.time_current = 0.0
    status.chkpt_number = 0
    status.chkpt_last = 0.0
    status.chkpt_interval = problem.cpi
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
            (mara.fluid.size / (wall_step + 1e-12)) * 1e-3,
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
    scheme = FishSolver()
    plot = PlottingOptions()

    parser.add_option("--dry-run", action="store_true",
                      help="just print run configuation and exit")
    args = [a for a in sys.argv if not a.startswith('-') and a != __file__]

    def add_option(g, k, v):
        k = k.replace('_', '-')
        if type(v) in [int, float, str]:
            g.add_option('--'+k, default=v, metavar=v, type=type(v))
        elif type(v) is bool:
            if v:
                g.add_option('--no-'+k, default=True, action='store_false',
                             metavar=v)
            else:
                g.add_option('--'+k, default=False, action='store_true',
                             metavar=v)
        elif type(v) is list:
            v = [str(vi) for vi in v]
            g.add_option('--'+k, default=','.join(v), metavar=','.join(v))

    if len(args) > 0:
        runparam = imp.load_source("runparam", args[0])
    else:
        runparam = imp.new_module("runparam")

    try:
        problem_cls = runparam.problem_cls
    except AttributeError:
        problem_cls = problems.get_problem_class()

    # get the options dict from configurable objects
    problem_cfg = problem_cls.options
    scheme_cfg = scheme.options
    plotting_cfg = plot.options

    # create corresponding options groups for command line parsing
    problem_opt = optparse.OptionGroup(parser, "problem settings for %s" %
                                       problem_cls.__name__)
    scheme_opt = optparse.OptionGroup(parser, "scheme settings")
    plotting_opt = optparse.OptionGroup(parser, "plotting settings")

    # populate those groups with default values
    for k,v in problem_cfg.items(): add_option(problem_opt, k, v)
    for k,v in scheme_cfg.items(): add_option(scheme_opt, k, v)
    for k,v in plotting_cfg.items(): add_option(plotting_opt, k, v)

    # add the option groups to the parser
    parser.add_option_group(problem_opt)
    parser.add_option_group(scheme_opt)
    parser.add_option_group(plotting_opt)
    opts, args = parser.parse_args()

    # get runtime config from runparam module and update confgurable objects
    for k,v in getattr(runparam, 'problem_cfg', { }).items():
        problem_cfg[k] = v
    for k,v in getattr(runparam, 'scheme_cfg', { }).items():
        scheme_cfg[k] = v
    for k,v in getattr(runparam, 'plotting_cfg', { }).items():
        plotting_cfg[k] = v


    # add command line options to configurable objects
    for k,v in opts.__dict__.items():
        if k in problem_cfg: problem_cfg[k] = v
        if k in scheme_cfg: scheme_cfg[k] = v
        if k in plotting_cfg: plotting_cfg[k] = v

    print "\n"
    print "run configuration:"
    print "------------------"
    print "\n  problem settings:"
    for k,v in [('problem name', problem_cls.__name__)] + problem_cfg.items():
        print "    %s: %s" % (k, v)
    print "\n  scheme settings:"
    for k,v in scheme_cfg.items():
        print "    %s: %s" % (k, v)
    print "\n  plotting settings:"
    for k,v in plotting_cfg.items():
        print "    %s: %s" % (k, v)
    print "\n"

    for k,v in scheme_cfg.items(): setattr(scheme, k, v)
    for k,v in plotting_cfg.items(): setattr(plot, k, v)

    if not opts.dry_run:
        problem = problem_cls(**problem_cfg)
        main(problem, scheme, plot)
