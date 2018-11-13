#! /usr/bin/env python3

""" Computes speedups from timing or bandwidth data
    Use `python3 computespeedup.py --help` to see all available options.
"""

import sys
import argparse
import numpy as np

def computescaling(filelist, cols, inputtype):
    for i in range(len(filelist)):
        filename = filelist[i]
        outfilename = filename.split('.')[-1]
        data = np.genfromtxt(filename)
        numsteps = data.shape[0]

        if inputtype == "time":
            for j in cols:
                data[:,j] = data[0,j]/data[:,j]
        else:
            for j in cols:
                data[:,j] = data[:,j]/data[0,j]

        np.savetxt(filename+".spdp", data, fmt="%10.5f")

if __name__ == "__main__":

    if(len(sys.argv) < 2):
        print("Error. Please provide input file name.")
        sys.exit(-1)

    parser = argparse.ArgumentParser(description="Converts wall times or bandwidths into speedups")
    parser.add_argument("files", nargs='+')
    parser.add_argument("--cols", nargs='+', type=int, help = "Columns in the input file to scale")
    parser.add_argument("--input_type", default="time", help = "'time' or 'rate', depending on whether\
             direct speedups or inverse speedups are needed")
    args = parser.parse_args(sys.argv)

    computescaling(args.files[1:], args.cols, args.input_type)

