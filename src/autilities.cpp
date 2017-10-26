#include "autilities.hpp"
#include <iostream>

namespace acfd {

std::ifstream open_file_toRead(const std::string file)
{
	std::ifstream fin(file);
	if(!fin) {
		std::cout << "! Could not open file "<< file <<" !\n";
		std::abort();
	}

	return fin;
}

std::ofstream open_file_toWrite(const std::string file)
{
	std::ofstream fout(file);
	if(!fout) {
		std::cout << "! Could not open file "<< file <<" !\n";
		//std::abort();
	}

	return fout;
}

}
