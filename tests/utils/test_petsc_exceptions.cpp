#undef NDEBUG
#define DEBUG 1

#include <stdexcept>
#include <iostream>
#include <cassert>

#include <petscsys.h>
#include <petscvec.h>

#include "utilities/aerrorhandling.hpp"

using namespace fvens;

int main(int argc, char *argv[])
{
    int ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);
    Vec x{};
    ierr = VecCreate(PETSC_COMM_WORLD, &x);
    bool encountered_exception = false;
    try {
        double *xarr{};
        ierr = VecGetArray(x, &xarr);
        petsc_throw(ierr, " In test");
    } catch (std::exception& e) {
        Petsc_exception &pe = dynamic_cast<Petsc_exception&>(e);
        encountered_exception = true;
        const std::string msg = pe.what();
        std::cout << "Error message:\n" << msg << std::endl;
        const auto pos = msg.find("No support for this operation for this object type");
        assert(pos != std::string::npos);
    }
    assert(encountered_exception);
    ierr = VecDestroy(&x);
    ierr = PetscFinalize(); CHKERRQ(ierr);
    return ierr;
}
