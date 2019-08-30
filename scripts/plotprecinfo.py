#! /usr/bin/env python3

""" Plots convergence histories.
    Use `python3 plotconv.py --help` to see all available options.
"""

import sys
import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def setAxisParams(ax, baseLineWidth):
    """ Sets line style and grid lines for a given pyplot axis."""
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which='major', axis='x', lw=0.5*baseLineWidth, linestyle='-', color='0.75')
    ax.grid(which='minor', axis='x', lw=0.5*baseLineWidth, linestyle=':', color='0.75')
    ax.grid(which='major', axis='y', lw=0.5*baseLineWidth, linestyle='-', color='0.75')
    ax.grid(which='minor', axis='y', lw=0.5*baseLineWidth, linestyle=':', color='0.75')

def plotquantity(filelist, quantname, numits, runs, labellist, opts, imageformatstring):
    plt.close()
    markdivisor = 10
    for i in range(len(filelist)):
        filenameprefix = filelist[i]

        xlabelstr = "Pseudo-time steps"
        ylabelstr = " "

        for j in range(1,runs+1):
            filename = filenameprefix + str(j) + ".conv"
            if not os.path.isfile(filename):
                break
            data = np.genfromtxt(filename)
            numsteps = data.shape[0]
            opts['markinterval'] = int(numsteps/markdivisor)

            # number of points to plot
            pltdatalen = len(data[:,0])-numits

            if quantname == "iluresidual":
                if j == 1:
                    plt.plot(np.log10(data[numits:,0]/data[numits:,1]), \
                            lw=opts['linewidth'], ls=opts['linetype'][i], color=opts['colorlist'][i], \
                            marker=opts['marklist'][i], ms=opts['marksize'], \
                            mew=opts['markedgewidth'], \
                            markevery=list(range(0,pltdatalen,opts['markinterval'])), \
                            label=labellist[i])
                else:
                    plt.plot(np.log10(data[numits:,0]/data[numits:,1]), \
                            lw=opts['linewidth'], ls=opts['linetype'][i], color=opts['colorlist'][i], \
                            marker=opts['marklist'][i], ms=opts['marksize'], \
                            mew=opts['markedgewidth'], \
                            markevery=list(range(0,pltdatalen,opts['markinterval'])))
                xlabelstr = "Pseudo-time steps"
                ylabelstr = "Log normalized ILU iteration residual 1-norm"
                #plt.gca().set_ylim([0,None])
            elif quantname == "ddom_min_upper":
                if j == 1:
                    plt.plot(np.log10(1.0-data[numits:,2]), \
                            lw=opts['linewidth'], ls=opts['linetype'][i], color=opts['colorlist'][i], \
                            marker=opts['marklist'][i], ms=opts['marksize'], \
                            mew=opts['markedgewidth'], \
                            markevery=list(range(0,pltdatalen,opts['markinterval'])), \
                            label=labellist[i])
                else:
                    plt.plot(np.log10(1.0-data[numits:,2]), \
                            lw=opts['linewidth'], ls=opts['linetype'][i], color=opts['colorlist'][i], \
                            marker=opts['marklist'][i], ms=opts['marksize'], \
                            mew=opts['markedgewidth'], \
                            markevery=list(range(0,pltdatalen,opts['markinterval'])))
                ylabelstr = "Log max-norm of Jacobi iteration matrix of U-factor"
                #plt.gca().set_ylim([0,None])
            elif quantname == "ddom_min_lower":
                if j == 1:
                    plt.plot(np.log10(1.0-data[numits:,4]), \
                            lw=opts['linewidth'], ls=opts['linetype'][i], color=opts['colorlist'][i], \
                            marker=opts['marklist'][i], ms=opts['marksize'], \
                            mew=opts['markedgewidth'], \
                            markevery=list(range(0,pltdatalen,opts['markinterval'])), \
                            label=labellist[i])
                else:
                    plt.plot(np.log10(1.0-data[numits:,4]), \
                            lw=opts['linewidth'], ls=opts['linetype'][i], color=opts['colorlist'][i], \
                            marker=opts['marklist'][i], ms=opts['marksize'], \
                            mew=opts['markedgewidth'], \
                            markevery=list(range(0,pltdatalen,opts['markinterval'])))
                ylabelstr = "Log max-norm of Jacobi iteration matrix of L-factor"
                #plt.gca().set_ylim([0,None])
            elif quantname == "cfl":
                if j == 1:
                    plt.plot(data[numits:,6], \
                            lw=opts['linewidth'], ls=opts['linetype'][i], color=opts['colorlist'][i], \
                            marker=opts['marklist'][i], ms=opts['marksize'], \
                            mew=opts['markedgewidth'], \
                            markevery=list(range(0,pltdatalen,opts['markinterval'])), \
                            label=labellist[i])
                else:
                    plt.plot(data[numits:,6], \
                            lw=opts['linewidth'], ls=opts['linetype'][i], color=opts['colorlist'][i], \
                            marker=opts['marklist'][i], ms=opts['marksize'], \
                            mew=opts['markedgewidth'], \
                            markevery=list(range(0,pltdatalen,opts['markinterval'])))
                ylabelstr = "CFL number"

    plt.xlabel(xlabelstr, fontsize="medium")
    plt.ylabel(ylabelstr, fontsize="medium")

    ax = plt.axes()
    setAxisParams(ax,opts['linewidth'])
    plt.legend(loc="best", fontsize="medium")

    plt.savefig(filename.split('/')[-1].split('.')[0]+"-"+quantname+"." + imageformatstring, dpi=200,\
            bbox_inches="tight")

if __name__ == "__main__":

    opts = { \
            "marklist" : ['.', 'x', '+', '^', 'v', '<', '>', 'd'],
            "colorlist" : ['c', 'b', 'r', 'g', 'o', 'm', 'k'],
            "linetype" : ['-', '--', '-.', ':', '--', '--', '--'],
            "linewidth" : 0.75,
            "marksize" : 5,
            "markedgewidth" : 1 \
            }

    if(len(sys.argv) < 2):
        print("Error. Please provide input file name.")
        sys.exit(-1)

    parser = argparse.ArgumentParser(description = "Plots BLASTed PCInfo history w.r.t. iterations. Note that we plot multiple runs of a setting using the same line type and symbol.")
    parser.add_argument("files", nargs='+', help="Prefixes of files to plot, not including the run index")
    parser.add_argument("--labels", nargs='+', help = "Legend strings")
    parser.add_argument("--labelstr", default="", help = "Common suffix for legend strings")
    parser.add_argument("--maxruns", type=int, help = "Max. number of repeated runs for any setting")
    parser.add_argument("--start_iter", type=int, default=0, help = "Iteration to start plotting from")
    parser.add_argument("--format", default="eps", help = "Output format")
    args = parser.parse_args(sys.argv)

    plotquantity(args.files[1:], "iluresidual", args.start_iter, args.maxruns, args.labels, opts, args.format)
    plotquantity(args.files[1:], "ddom_min_upper", args.start_iter, args.maxruns, args.labels, opts, args.format)
    plotquantity(args.files[1:], "ddom_min_lower", args.start_iter, args.maxruns, args.labels, opts, args.format)
    plotquantity(args.files[1:], "cfl", args.start_iter, args.maxruns, args.labels, opts, args.format)

