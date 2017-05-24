#! /bin/bash
#PBS -l nodes=1:ppn=16
#PBS -l walltime=00:40:00
#PBS -A rck-371-aa
#PBS -o expl.log
#PBS -e expl.err
#PBS -N expl

# to be executed from Euler1d-acc/runs/ or equivalent

# ppn=16:sandybridge will always be run on a SW2 sandy bridge node, according to http://www.hpc.mcgill.ca/index.php/starthere/81-doc-pages/91-guillimin-job-submit

cd $PBS_O_WORKDIR
module load intel/2017.00

export OMP_NUM_THREADS=8
time ../build/fvense ../testcases/2dcylnder/expliciteuler.control

