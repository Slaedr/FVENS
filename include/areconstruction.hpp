/** @file areconstruction.hpp
 * @brief Classes for different gradient reconstruction schemes.
 * @author Aditya Kashi
 * @date February 3, 2016
 */

using namespace std;

namespace acfd
{

/// Abstract class for variable gradient reconstruction schemes
class Reconstruction
{
	UTriMesh* m;
	/// Cell centers' x-coords
	Matrix<double>* xc;
	/// Cell centers' y-coords
	Matrix<double>* yc;
	/// Ghost cell centers' x-coords
	Matrix<double>* xcg;
	/// Ghost cell centers' y-coords
	Matrix<double>* ycg;
	/// Number of converved variables
	int nvars;
	/// Cell-centered flow vaiables
	Matrix<double>* u;
	/// Cell-centred x-gradients
	Matrix<double>* dudx;
	/// Cell-centred y-gradients
	Matrix<double>* dudy;

public:
	void setup(UTriMesh* mesh, Matrix<double>* unk, Matrix<double>* gradx, Matrix<double>* grady, Matrix<double>* _xc, Matrix<double>* _yc, Matrix<double>* _xcg, Matrix<double>* _ycg);
	virtual void compute_gradients() = 0;
};

Reconstruction::setup(UTriMesh* mesh, Matrix<double>* unk, Matrix<double>* gradx, Matrix<double>* grady, Matrix<double>* _xc, Matrix<double>* _yc, Matrix<double>* _xcg, Matrix<double>* _ycg)
{
	m = mesh;
	u = unk;
	dudx = gradx;
	dudy = grady;
	xc = _xc;
	yc = _yc;
	xcg = _xcg;
	ycg = _ycg;
	nvars = u->cols();
}

/**
 * @brief Implements reconstruction using the Green-Gauss theorem over elements.
 */
class GreenGaussReconstruction : public Reconstruction
{
public:
	void compute_gradients();
};

GreenGaussReconstruction::compute_gradients()
{
	for(int i = 0; i < u->cols(); i++)
}

/// Class implementing linear weighted least-squares reconstruction
class WeightedLeastSquaresReconstruction : public Reconstruction
{
	Matrix<double> w2x2;
	Matrix<double> w2y2;
	Matrix<double> w2xy;
	Matrix<double> w2xu;
	Matrix<double> w2yu;
};

} // end namespace
