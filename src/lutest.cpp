#include <alinalg.hpp>

using namespace amat;
using namespace acfd;
using namespace std;

int main()
{
	Matrix<double> A(4,4);
	A(0,0) = 1;		A(0,1) = 2.0;	A(0,2) = 0.5;	A(0,3) = 0;
	A(1,0) = 6.0;	A(1,1) = 0.0;	A(1,2) = 1.5;	A(1,3) = 2.5;
	A(2,0) = 2.5;	A(2,1) = 5.0;	A(2,2) = -6.2;	A(2,3) = 1.0;
	A(3,0) = 2.0;	A(3,1) = 1.1;	A(3,2) = 5.5;	A(3,3) = 3.2;

	Matrix<double> b(4,1);
	b(0) = 6.5;		b(1) = 20.5;	b(2) = -2.1;		b(3) = 33.5;

	// with above A and b, answer should be [1 2 3 4]

	Matrix<int> p(4,1);
	Matrix<double> x(4,1);

	LUfactor(A,p);
	LUsolve(A,p,b,x);

	x.mprint();

	cout << endl;
	return 0;
}
