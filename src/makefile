#GPP = g++
INC_DIR = ../include/
	
eulerl:
	$(GPP) eulerdr.cpp -o euler -std=c++11 -ggdb -I$(INC_DIR)

eulerm:
	$(GPP) euler-muscl-dr.cpp -o eulerm.out -std=c++11 -ggdb -I$(INC_DIR)
eulermp:
	$(GPP) euler-muscl-dr.cpp -o eulermp.out -fopenmp -std=c++11 -ggdb -I$(INC_DIR)

