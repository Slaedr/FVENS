#! /usr/bin/env python3

""" Plots strong scaling for the BLASTed perftest scaling reports as input.
    Use `python3 plotstrongscale.py --help` to see all available options.
"""

import sys
import argparse
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MaxNLocator

def setAxisParams(ax, baseLineWidth):
    """ Sets line style and grid lines for a given pyplot axis."""
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which='major', axis='x', lw=0.5*baseLineWidth, linestyle='-', color='0.75')
    ax.grid(which='minor', axis='x', lw=0.5*baseLineWidth, linestyle=':', color='0.75')
    ax.grid(which='major', axis='y', lw=0.5*baseLineWidth, linestyle='-', color='0.75')
    ax.grid(which='minor', axis='y', lw=0.5*baseLineWidth, linestyle=':', color='0.75')

def plotstrongscaling(filelist, basethreads, ht, phase, labellist, labelstr, titlestr, opts, imageformatstring):
    plt.close()
    fig = plt.figure()
    markdivisor = 20
    maxspeedup = 1.0

    phasestring = ""
    ylabelstr = ""
    plotcol = 0
    if phase == "factor":
        plotcol = 3
        phasestring = "factorization"
        ylabelstr = "Speedup x " + str(basethreads)
    elif phase == "apply":
        plotcol = 4
        phasestring = "application"
        ylabelstr = "Speedup x " + str(basethreads)
    elif phase == "all":
        plotcol = 5
        ylabelstr = "Speedup x " + str(basethreads)
    elif phase == "liniters":
        plotcol = 8
        ylabelstr = "Total linear solver iterations"
    else:
        print("Invalid phase!")
        return

    for i in range(len(filelist)):
        filename = filelist[i]
        data = np.genfromtxt(filename)

        if phase == "liniters":
            fulldata = np.zeros((data.shape[0],data.shape[1]))
            fulldata[:,:] = data[:,:]
        else:
            fulldata = np.zeros((data.shape[0]+1,data.shape[1]))
            fulldata[0,0] = basethreads
            fulldata[0,3:6] = 1.0
            fulldata[1:,:] = data[:,:]
        
        if maxspeedup < fulldata[-1,plotcol]*basethreads:
            maxspeedup = fulldata[-1,plotcol]*basethreads

        plt.plot(fulldata[:,0]/ht, fulldata[:,plotcol]*basethreads, \
                lw=opts['linewidth'], ls=opts['linetype'][i], color=opts['colorlist'][i], \
                marker=opts['marklist'][i], ms=opts['marksize'], \
                mew=opts['markedgewidth'], \
                label=labellist[i]+labelstr)

    # Note that we now just use data from the last file to be processed in the list of files
    if phase != "liniters":
        plt.plot(fulldata[:,0]/ht, fulldata[:,0]/ht, lw=opts['linewidth']+0.1, ls=':', label="Ideal")
    if maxspeedup < fulldata[-1,0]/ht:
        maxspeedup = fulldata[-1,0]/ht
    plt.xlabel("Number of cores", fontsize="medium")
    plt.ylabel(ylabelstr, fontsize="medium")

    ax = plt.axes()
    if phase != "liniters":
        ymin = basethreads
        ymax = maxspeedup+0.5
        #ax.set_ymargin(0.05)
        ax.set_yticks(np.arange(ymin,ymax,round((ymax-ymin)/10)))
    setAxisParams(ax,opts['linewidth'])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)
    plt.legend(loc="best", fontsize="medium")
    if titlestr != "":
        fig.suptitle(titlestr, fontsize="medium")

    plt.savefig(filename.split('/')[-1].split('.')[0] + "-" + phasestring + "-speedup"+"." \
            + imageformatstring, dpi=150)

if __name__ == "__main__":

    opts = { \
            "marklist" : ['.', 'x', '+', '^', 'v', '<', '>', 'd'],
            "colorlist" : ['k', 'b', 'r', 'g', 'c', 'm', 'k'],
            "linetype" : ['-', '--', '-.', '--', '--'],
            "linewidth" : 0.75,
            "marksize" : 5,
            "markedgewidth" : 1 \
            }

    if(len(sys.argv) < 2):
        print("Error. Please provide input file name.")
        sys.exit(-1)

    parser = argparse.ArgumentParser(description="Plots residual history w.r.t. iterations and \
            wall time starting at a specified iteration")
    parser.add_argument("files", nargs='+')
    parser.add_argument("--basethreads", default=1, type=int, help = "Number of threads for base case")
    parser.add_argument("--ht", default=1, type=int, help = "Number of hyper-threads used per core")
    parser.add_argument("--phase", help = "Phase of preconditioner to consider (factor, apply, all)")
    parser.add_argument("--labels", nargs='+', help = "Legend strings")
    parser.add_argument("--labelstr", default="", help = "Common suffix for legend strings")
    parser.add_argument("--title", default="", help = "Title string for the plot")
    parser.add_argument("--format", default="png", help = "Output format")
    args = parser.parse_args(sys.argv)

    plotstrongscaling(args.files[1:], args.basethreads, args.ht, args.phase, args.labels, args.labelstr,
            args.title, opts, args.format)

