/** @file areconstruction.hpp
 * @brief Classes for different gradient reconstruction schemes.
 * @author Aditya Kashi
 * @date February 3, 2016
 */

using namespace std;

namespace acfd
{

class Reconstruction
{
	UTriMesh* m;
	int nvars;
	/// Cell-centered flow vaiables
	Matrix<double>* u;
	/// Cell-centred x-gradients
	Matrix<double>* dudx;
	/// Cell-centred y-gradients
	Matrix<double>* dudy;

public:
	virtual void compute_gradients() = 0;
};
 
class GreenGaussReconstruction : public Reconstruction
{
public:
	void setup(UTriMesh* mesh, Matrix<double>* unk, Matrix<double>* gradx, Matrix<double>* grady);
	void compute_gradients();
};

GreenGaussReconstruction::setup(UTriMesh* mesh, Matrix<double>* unk, Matrix<double>* gradx, Matrix<double>* grady)
{
	m = mesh;
	u = unk;
	dudx = gradx;
	dudy = grady;
}

GreenGaussReconstruction::compute_gradients()
{
	for(int i = 0; i < u->cols(); i++)
}

} // end namespace
