PREFIX = ../testcases/
#GPP = g++
INC_DIR = .
OPTIONS = -std=c++11 -ggdb
	
euler:
	$(GPP) eulerdr.cpp -o $(PREFIX)euler.out $(OPTIONS) -I$(INC_DIR)

fvens:
	$(GPP) $(OPTIONS) -I$(INC_DIR) explicitsolver.cpp -o $(PREFIX)fvens.out

fvensp:
	$(GPP) -fopenmp $(OPTIONS) -I$(INC_DIR) explicitsolver.cpp -o $(PREFIX)fvensp.out

