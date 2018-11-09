CXXFLAGS := -O1 -std=c++14 -g

all : runmesh3dtest

mesh3d: ../../src/mesh/amesh.hpp ../../src/utilities/aarray2d.hpp ../../src/aconstants.hpp ../../src/utilities/aoptionparser.hpp
	$(CXX) $(CXXFLAGS) -c ../../src/mesh/amesh.cpp ../../src/utilities/aarray2d.cpp ../../src/utilities/aoptionparser.cpp -I../../src/ -I$(EIGEN3_ROOT) -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include

mesh3dtest: mesh3d
	$(CXX) $(CXXFLAGS) -c mesh3dtest.cpp
	$(CXX) $(CXXFLAGS) -o mesh3dtest mesh3dtest.o amesh.o aarray2d.o -lboost_program_options -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc

runmesh3dtest: mesh3dtest
	./mesh3dtest
