#include <amatrix2.hpp>

namespace amat {

double dabs(double x)
{
	if(x < 0) return (-1.0)*x;
	else return x;
}
double minmod(double a, double b)
{
	if(a*b>0 && dabs(a) <= dabs(b)) return a;
	else if (a*b>0 && dabs(b) < dabs(a)) return b;
	else return 0.0;
}

}
