#ifndef TEST_H
#define TEST_H

#include <cmath>
#include <limits>

#define TASSERT(x) if(!(x)) return -1

/**
 * Check relative or absolute difference based on the magnitude.
 */
template <typename scalar>
inline bool close_enough(const scalar& value, const scalar& ref,
                         const double tolerance) {
    constexpr double tol = 1e-8;
    if(ref < tol) {
        return std::abs(value - ref) < tolerance;
    } else {
        return std::abs(value - ref) / std::abs(ref) < tolerance;
    }
}

#endif
