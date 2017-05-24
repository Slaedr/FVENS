#! /bin/bash
#PBS -l nodes=1:ppn=16:ivybridge
#PBS -l walltime=00:30:00
#PBS -A rck-371-aa
#PBS -o 2dcylinder-vfine-impl.log
#PBS -e 2dcylinder-vfine-impl.err
#PBS -N 2dcylinder

# to be executed from Euler1d-acc/runs/ or equivalent

# ppn=16:sandybridge will always be run on a SW2 sandy bridge node, according to http://www.hpc.mcgill.ca/index.php/starthere/81-doc-pages/91-guillimin-job-submit

threadseq='1 2 4 8 10 16'

cd $PBS_O_WORKDIR

for i in ${threadseq}; do

	export OMP_NUM_THREADS=$i
	time ../build/steadyfvensi ../testcases/2dcylnder/vfineimplicit.control

done
