CXXFLAGS := -O1 -std=c++14 -g

all : mesh3dtest

mesh3d: ../../src/mesh/amesh.hpp ../../src/utilities/aarray2d.hpp ../../src/aconstants.hpp ../../src/utilities/aoptionparser.hpp ../../src/utilities/aerrorhandling.hpp
	$(CXX) $(CXXFLAGS) -c ../../src/mesh/amesh.cpp ../../src/utilities/aarray2d.cpp ../../src/utilities/aoptionparser.cpp ../../src/utilities/aerrorhandling.cpp -I../../src/ -I$(EIGEN3_ROOT) -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include

mesh3dtest: mesh3d
	$(CXX) $(CXXFLAGS) -c mesh3dtest.cpp -I../../src -I$(EIGEN3_ROOT) -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include
	$(CXX) $(CXXFLAGS) -o mesh3dtest mesh3dtest.o amesh.o aoptionparser.o aarray2d.o aerrorhandling.o -lboost_program_options -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc

check:
	./mesh3dtest

clean:
	rm -f *.o mesh3dtest
