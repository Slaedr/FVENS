PREFIX = ../testcases/
#GPP = g++
INC_DIR = .
OPTIONS = -std=c++11 -ggdb -O3
	
euler:
	$(GPP) eulerdr.cpp -o $(PREFIX)euler.out $(OPTIONS) -I$(INC_DIR)

fvens:
	$(GPP) $(OPTIONS) -I$(INC_DIR) amatrix2.cpp adatastructures.cpp amesh2dh.cpp aoutput.cpp explicitsolver.cpp -o $(PREFIX)fvens.out

fvensp:
	$(GPP) -fopenmp $(OPTIONS) -I$(INC_DIR) explicitsolver.cpp -o $(PREFIX)fvensp.out

conv:
	$(GPP) $(OPTIONS) -I$(INC_DIR) adatastructures.cpp amesh2dh.cpp aoutput.cpp utilities/convertformat.cpp -o $(PREFIX)conv.out

