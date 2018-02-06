#include <iostream>
#include <string>
#include "../amesh2dh.hpp"
#include "../aoutput.hpp"

using namespace amat;
using namespace acfd;
using namespace std;

int main(int argc, char* argv[])
{
	if(argc < 4) {
		cout << "Need: 1. Input mesh file, 2. Output mesh file 3. Output format.\n" << endl;
	}
	string confilename(argv[1]);
	string inmesh = argv[1], outmesh = argv[2], outformat = argv[3];

	//cout << "Input file is of type " << informat << ". Writing as " << outformat << ".\n";

	UMesh2dh m;
	m.readMesh(inmesh);

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
