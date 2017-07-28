#include "amesh2dh.hpp"

using namespace amat;
using namespace acfd;
using namespace std;

int main()
{
	UMesh2dh m;
	m.readGmsh2("../data/testhybrid.msh",2);
	m.compute_topological();
	
	int naface = 15;
	int face = m.gelemface(18-naface,3);
	cout << m.gintfac(face,2) << " " << m.gintfac(face,3) << endl;
	
	cout << endl;
	return 0;
}
