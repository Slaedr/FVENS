#! /bin/bash
#PBS -l nodes=1:ppn=2:sandybridge
#PBS -l walltime=00:10:00
#PBS -A rck-371-aa
#PBS -o 2dcylnder-fine-1thread.log
#PBS -e 2dcylinder.err
#PBS -N 2dcylinder

# to be executed from Euler1d-acc/runs/ or equivalent

# ppn=16:sandybridge will always be run on a SW2 sandy bridge node, according to http://www.hpc.mcgill.ca/index.php/starthere/81-doc-pages/91-guillimin-job-submit

export OMP_NUM_THREADS=1

cd $PBS_O_WORKDIR
#cat /proc/cpuinfo | grep 'model name'
time ../build/bin/fvense ../testcases/2dcylnder/expliciteuler.control
