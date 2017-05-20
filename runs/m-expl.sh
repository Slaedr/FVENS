#! /bin/bash
#PBS -l nodes=1:ppn=16:ivybridge
#PBS -l walltime=00:40:00
#PBS -A rck-371-aa
#PBS -o 2dcylnder-vfine-expl.log
#PBS -e 2dcylinder.err
#PBS -N 2dcylinder

# to be executed from Euler1d-acc/runs/ or equivalent

# ppn=16:sandybridge will always be run on a SW2 sandy bridge node, according to http://www.hpc.mcgill.ca/index.php/starthere/81-doc-pages/91-guillimin-job-submit

cd $PBS_O_WORKDIR

export OMP_NUM_THREADS=1
time ../build/fvense ../testcases/2dcylnder/expliciteuler.control

export OMP_NUM_THREADS=2
time ../build/fvense ../testcases/2dcylnder/expliciteuler.control

export OMP_NUM_THREADS=4
time ../build/fvense ../testcases/2dcylnder/expliciteuler.control

export OMP_NUM_THREADS=8
time ../build/fvense ../testcases/2dcylnder/expliciteuler.control

export OMP_NUM_THREADS=10
time ../build/fvense ../testcases/2dcylnder/expliciteuler.control

export OMP_NUM_THREADS=16
time ../build/fvense ../testcases/2dcylnder/expliciteuler.control
