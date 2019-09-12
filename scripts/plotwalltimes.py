#! /usr/bin/env python3

""" Plots strong scaling for the BLASTed perftest scaling reports as input.
    Use `python3 plotstrongscale.py --help` to see all available options.
"""

import sys
import argparse
import re
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

def getBaseData(filename):
    """ Returns base data (raw timings of the 1-thread run) of a FVENS threads report file.
    """
    ff = open(filename,'r')
    alllines = ff.readlines()
    ff.close()
    
    baseline = alllines[2]
    basestrings = baseline.split()[1:]
    print("Timing line is " + str(basestrings))
    assert(len(basestrings) == 19)
    
    firststrings = alllines[6].split()[1:]
    print("Base case line is " + str(firststrings))
    assert(len(firststrings) == 13)

    bd = np.zeros(13)
    bd[0] = int(firststrings[0])
    for i in range(3,9):
        bd[i] = float(firststrings[i])
    for i in range(9,12):
        bd[i] = int(firststrings[i])
    bd[12] = float(firststrings[12])

    bd[3] = float(re.split('[,;]', basestrings[9])[0])
    bd[4] = float(re.split('[,;]', basestrings[13])[0])
    bd[5] = float(re.split('[,;]', basestrings[5])[0])
    #bd[4] = float(basestrings[13])
    #bd[5] = float(basestrings[5])

    print(" Base line of file " + filename.split('/')[-1] + "is " + str(bd))
    return bd

def plotwalltimes(filelist, ht, phase, labellist, labelstr, titlestr, opts, imageformatstring):
    plt.close()
    fig = plt.figure()
    markdivisor = 20
    maxspeedup = 1.0

    phasestring = ""
    ylabelstr = "Log (base 10) of wall time (log seconds)"
    plotcol = 0
    if phase == "factor":
        plotcol = 3
        phasestring = "factorization"
    elif phase == "apply":
        plotcol = 4
        phasestring = "application"
    elif phase == "all":
        plotcol = 5
    elif phase == "liniters":
        plotcol = 8
    elif phase == "nliters":
        plotcol = 10
    else:
        print("Invalid phase!")
        return

    for i in range(len(filelist)):
        filename = filelist[i]
        data = np.genfromtxt(filename)

        # Get base line
        bsl = getBaseData(filename)

        fulldata = np.zeros((data.shape[0]+1,data.shape[1]))
        fulldata[0,:] = bsl
        fulldata[1:,0] = data[:,0]
        if phase == "liniters" or phase == "nliters":
            fulldata[1:,:] = data[:,:]
        else:
            fulldata[1:,1:] = bsl[1:]/data[:,1:]
        
        plt.plot(fulldata[:,0]/ht, np.log10(fulldata[:,plotcol]), \
                lw=opts['linewidth'], ls=opts['linetype'][i], color=opts['colorlist'][i], \
                marker=opts['marklist'][i], ms=opts['marksize'], \
                mew=opts['markedgewidth'], \
                label=labellist[i]+labelstr)

    # Note that we now just use data from the last file to be processed in the list of files
    if maxspeedup < fulldata[-1,0]/ht:
        maxspeedup = fulldata[-1,0]/ht
    plt.xlabel("Number of cores", fontsize="medium")
    plt.ylabel(ylabelstr, fontsize="medium")

    ax = plt.axes()
    #if phase != "liniters" and phase != "nliters":
        #ymin = basethreads
        #ymax = maxspeedup+0.5
        ##ax.set_ymargin(0.05)
        #ax.set_yticks(np.arange(ymin,ymax,round((ymax-ymin)/10)))
    setAxisParams(ax,opts['linewidth'])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)
    plt.legend(loc="best", fontsize="medium")
    if titlestr != "":
        fig.suptitle(titlestr, fontsize="medium")

    plt.savefig(filename.split('/')[-1].split('.')[0] + "-" + phasestring + "-walltimes"+"." \
            + imageformatstring, dpi=150, bbox_inches='tight')

if __name__ == "__main__":

    opts = { \
            "marklist" : ['.', 'x', '+', '^', 'v', '<', '>', 'd'],
            "colorlist" : ['k', 'b', 'r', 'g', 'c', 'm', 'k'],
            "linetype" : ['-', '--', '-.', '--', '--','-.'],
            "linewidth" : 0.75,
            "marksize" : 5,
            "markedgewidth" : 1 \
            }

    if(len(sys.argv) < 2):
        print("Error. Please provide input file name.")
        sys.exit(-1)

    parser = argparse.ArgumentParser(description="Plots wall-clock times of threaded preconditioners")
    parser.add_argument("files", nargs='+')
    #parser.add_argument("--basethreads", default=1, type=int, help = "Number of threads for base case")
    parser.add_argument("--ht", default=1, type=int, help = "Number of hyper-threads used per core")
    parser.add_argument("--phase", help = "Phase of preconditioner to consider (factor, apply, all,liniters,nliters)")
    parser.add_argument("--labels", nargs='+', help = "Legend strings")
    parser.add_argument("--labelstr", default="", help = "Common suffix for legend strings")
    parser.add_argument("--title", default="", help = "Title string for the plot")
    parser.add_argument("--format", default="eps", help = "Output format")
    args = parser.parse_args(sys.argv)

    plotwalltimes(args.files[1:], args.ht, args.phase, args.labels, args.labelstr,
                  args.title, opts, args.format)

