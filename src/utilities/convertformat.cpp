#include <amesh2dh.hpp>
#include <aoutput.hpp>

using namespace amat;
using namespace acfd;
using namespace std;

int main(int argc, char* argv[])
{
	if(argc < 2) {
		cout << "Need a control file!" << endl;
	}
	string confilename(argv[1]);
	ifstream conf(confilename);
	string dum, inmesh, informat, outmesh, outformat;
	conf >> dum; conf >> inmesh;
	conf >> dum; conf >> informat;
	conf >> dum; conf >> outmesh;
	conf >> dum; conf >> outformat;
	conf.close();

	cout << "Input file is of type " << informat << ". Writing as " << outformat << ".\n";

	UMesh2dh m;
	if(informat == "domn")
		m.readDomn(inmesh);
	else if(informat == "msh")
		m.readGmsh2(inmesh,2);
	else {
		cout << "Invalid format. Exiting." << endl;
		return -1;
	}

	if(outformat == "msh")	
		m.writeGmsh2(outmesh);
	else if(outformat == "vtu")
		writeMeshToVtu(outmesh, m);
	else {
		cout << "Invalid format. Exiting." << endl;
		return -1;
	}

	cout << endl;
	return 0;
}
