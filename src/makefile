#GPP = g++
INC_DIR = ../include/
OPTIONS = -std=c++11 -ggdb
	
euler:
	$(GPP) eulerdr.cpp -o euler.out -std=c++11 -ggdb -I$(INC_DIR)

eulerm:
	$(GPP) euler-muscl-dr.cpp -o eulerm.out $(OPTIONS) -I$(INC_DIR)

eulermp:
	$(GPP) euler-muscl-dr.cpp -o eulermp.out -fopenmp $(OPTIONS) -I$(INC_DIR)

