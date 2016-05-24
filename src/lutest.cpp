#include <alinalg.hpp>

using namespace amat;
using namespace acfd;
using namespace std;

int main()
{
	// in the following two tests, in-situ LU factorization works fine.
	
	Matrix<double> A(4,4);
	A(0,0) = 11;	A(0,1) = 9;		A(0,2) = 24;	A(0,3) = 2;
	A(1,0) = 1;		A(1,1) = 5;		A(1,2) = 2;		A(1,3) = 6;
	A(2,0) = 3;		A(2,1) = 17;	A(2,2) = 18;	A(2,3) = 1;
	A(3,0) = 2;		A(3,1) = 5;		A(3,2) = 7;		A(3,3) = 1;
	Matrix<double> Ao = A;

	Matrix<double> b(4,1);
	b(0) = 6.5;		b(1) = 20.5;	b(2) = -2.1;		b(3) = 3.5;

	Matrix<int> p(4,1);
	Matrix<double> x(4,1);
	
	/*Matrix<double> A(3,3);
	A(0,0) = 1;		A(0,1) = 3;		A(0,2) = 5;
	A(1,0) = 2;		A(1,1) = 4;		A(1,2) = 7;	
	A(2,0) = 1;		A(2,1) = 1;		A(2,2) = 0;
	Matrix<double> Ao = A;

	Matrix<double> b(4,1);
	b(0) = 6.5;		b(1) = 2.5;	b(2) = -2.1;

	Matrix<int> p(3,1);
	Matrix<double> x(3,1);*/

	LUfactor(A,p);
	A.mprint();
	p.mprint();
	LUsolve(A,p,b,x);

	cout << "The solution ";
	x.mprint();
	
	cout << "error ";
	Matrix<double> err = Ao*x-b;
	cout << ": " << err.l2norm() << endl;

	cout << endl;
	return 0;
}
