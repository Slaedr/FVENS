#! /bin/bash

cmake -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DSKYLAKE=1 -DCMAKE_BUILD_TYPE=Release $@
