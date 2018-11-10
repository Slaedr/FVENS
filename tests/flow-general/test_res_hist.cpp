/** \file
 * \brief Reads a FVENS residual history output file and does some sanity checks.
 */

#undef NDEBUG

#include <fstream>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <string>
#include <limits>

#define SMALL_TIME_NUMBER 0.001
#define SMALL_DEVIATION_NUMBER 1e-6

int main(int argc, char *argv[])
{
	assert(argc >= 2);

	const std::string outprefix = argv[1];
	std::ifstream fin(outprefix+"-residual_history.log");
	assert(fin);

	std::string line;

	// check header portion
	std::getline(fin,line);
	assert(line[0] == '#');
	std::getline(fin,line);
	assert(line[0] == '#');

	// main data
	int rows = 0;
	while(std::getline(fin,line)) {
		assert(line[0] != '#');
		rows++;
		int nums[2]; float dnums[6];
		sscanf(line.c_str(), "%d %f %f %f %f %d %f",
				&nums[0], &dnums[0], &dnums[1], &dnums[2], &dnums[3], &nums[1], &dnums[4]);

		assert(nums[0] == rows);                   // step
		assert(dnums[0] <= 0);                     // log rel residual
		assert(dnums[2] > SMALL_DEVIATION_NUMBER); // ODE wall time
		assert(dnums[3] >= 0);                     // Linear solver time
		assert(nums[1] >= 0);                      // Linear solver iters
		assert(dnums[4] >= 0.2);                   // CFL - arbitrary
	}

	assert(rows > 5);  // arbitrarily, at least 5 time steps

	fin.close();
	return 0;
}
