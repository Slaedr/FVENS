/** \file
 * \brief Reads a FVENS async threads speedup output file and does some sanity checks.
 */

#undef NDEBUG

#include <fstream>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <string>
#include <limits>

#define SMALL_TIME_NUMBER 0.001
#define SMALL_DEVIATION_NUMBER 1e-8

int main(int argc, char *argv[])
{
	assert(argc >= 2);
	//constexpr double meps = std::numeric_limits<double>::epsilon();

	const std::string outprefix = argv[1];
	std::ifstream fin(outprefix+".perf");

	std::string line;

	// check header portion
	bool headerlinefound = false;
	while(std::getline(fin,line)) 
	{
		if(line[0] != '#')
			break;
		const size_t found = line.find("threads");
		if(found != std::string::npos) 
		{
			headerlinefound = true;

			std::getline(fin,line);
			std::getline(fin,line);

			int nums[11]; double time[3]; char dm;
			sscanf(line.c_str(), "%c %d %d %d %d %d %d %lf %lf %d %d %d %d %lf", 
					&dm, &nums[0], &nums[1], &nums[2], &nums[3], &nums[4], &nums[5], &time[0],
					&time[1], &nums[7], &nums[8], &nums[9], &nums[10], &time[2]);
			assert(dm == '#');
			for(int i = 0; i < 6; i++)
				assert(nums[i] == 1);
			assert(time[0] > SMALL_DEVIATION_NUMBER);                       // deviation
			std::cout << "Nums 7 = " << nums[7] << std::endl;
			assert(nums[7] > 1);
			assert(nums[8] >= 1);
			assert(nums[9] >= 1);
			assert(time[1] > SMALL_TIME_NUMBER);
			assert(time[2] == 1.0);
			break;
		}
	}
	assert(headerlinefound);

	assert(std::getline(fin,line));
	assert(line[0] == '#');

	// main data
	int rows = 0;
	while(std::getline(fin,line)) {
		rows++;
		int nums[7]; double dnums[6];
		sscanf(line.c_str(), "%d %d %d %lf %lf %lf %lf %lf %d %d %d %d %lf", 
				&nums[0], &nums[1], &nums[2], &dnums[0], &dnums[1], &dnums[2], &dnums[3], &dnums[4],
				&nums[3], &nums[4], &nums[5], &nums[6], &dnums[5]);
		assert(nums[0] == 2 || nums[0] == 4 || nums[0] == 6);       // num threads
		assert(nums[1] == 1 || nums[1] == 2);                       // num build sweeps
		assert(nums[2] == 1 || nums[2] == 3);                       // apply sweeps

		for(int i = 0; i < 3; i++) {
			assert(dnums[i] > 0.05);                 // some speedup for build, apply, total
			assert(dnums[i] < 5.0);                  // assuming max 4 threads
		}

		assert(dnums[3] > SMALL_DEVIATION_NUMBER);   // should be some nonzero total deviation
		assert(dnums[4] > 0);                        // CPU time

		assert(nums[3] > 1);
		assert(nums[4] >= 1);
		assert(nums[5] > 1);

		// avg num iters should be roughly total iters by num steps (say 20% error)
		assert(std::abs(nums[4]*nums[5] - nums[3])/nums[3] < 0.2);        
		assert(std::abs(nums[4]*nums[5] -nums[3])/nums[3] < 0.2);

		assert(nums[6] == 1);                        // converged

		assert(dnums[5] > 0.01);                     // some NL speedup
		assert(dnums[5] < 5);                        // Assuming we use at most 4 threads for the test
	}

	assert(rows == 4);

	fin.close();

	// Check if convergence history output files can be opened
	std::ifstream conv;

	conv.open(outprefix+"-sweeps1_1-threads2.conv");
	if(!conv)
		throw std::ios_base::failure("File not found");
	conv.close();
	conv.open(outprefix+"-sweeps1_1-threads4.conv");
	if(!conv)
		throw std::ios_base::failure("File not found");
	conv.close();
	conv.open(outprefix+"-sweeps2_3-threads2.conv");
	if(!conv)
		throw std::ios_base::failure("File not found");
	conv.close();
	conv.open(outprefix+"-sweeps2_3-threads4.conv");
	if(!conv)
		throw std::ios_base::failure("File not found");
	conv.close();

	return 0;
}
