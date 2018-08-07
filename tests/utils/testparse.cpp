#undef NDEBUG
#define DEBUG 1

#include <iostream>
#include <cassert>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <petscsys.h>
#include "utilities/aoptionparser.hpp"
#include "utilities/controlparser.hpp"

using namespace std;
using namespace fvens;
namespace po = boost::program_options;

FlowParserOptions parse_solution_file(std::ifstream& inf)
{
	FlowParserOptions pd;
	pd.bcconf.resize(2);
	a_real tempalpha;
	inf >> pd.meshfile >> pd.vtu_output_file >> pd.logfile >> pd.lognres
	    >> pd.flowtype >> pd.gamma >> tempalpha >> pd.Minf;
	pd.bcconf[0].bc_type = SLIP_WALL_BC; inf >> pd.bcconf[0].bc_tag;
	pd.bcconf[1].bc_type = FARFIELD_BC; inf >> pd.bcconf[1].bc_tag;
	inf >> pd.num_out_walls >> pd.surfnameprefix >> pd.vol_output_reqd
	    >> pd.sim_type >> pd.invflux >> pd.gradientmethod >> pd.limiter
	    >> pd.pseudotimetype >> pd.initcfl >> pd.endcfl >> pd.tolerance >> pd.maxiter
	    >> pd.firstinitcfl >> pd.firstendcfl >> pd.firsttolerance >> pd.firstmaxiter;
	pd.alpha = tempalpha * PI/180.0;
	return pd;
}

void compare(const FlowParserOptions& opts, const FlowParserOptions& exsol)
{
	assert(opts.meshfile == exsol.meshfile);
	assert(opts.vtu_output_file == exsol.vtu_output_file);
	assert(opts.logfile == exsol.logfile);
	assert(opts.lognres == exsol.lognres);
	assert(opts.flowtype == exsol.flowtype);
	assert(opts.gamma == exsol.gamma);
	assert(opts.alpha == exsol.alpha);
	assert(opts.Minf == exsol.Minf);
	assert(opts.bcconf[0].bc_tag == exsol.bcconf[0].bc_tag);
	assert(opts.bcconf[1].bc_tag == exsol.bcconf[1].bc_tag);
	assert(opts.num_out_walls == exsol.num_out_walls);
	assert(opts.surfnameprefix == exsol.surfnameprefix);
	assert(opts.vol_output_reqd == exsol.vol_output_reqd);
	assert(opts.sim_type == exsol.sim_type);
	assert(opts.invflux == exsol.invflux);
	assert(opts.gradientmethod == exsol.gradientmethod);
	assert(opts.limiter == exsol.limiter);
	assert(opts.pseudotimetype ==exsol.pseudotimetype);
	assert(opts.initcfl		 ==exsol.initcfl);
	assert(opts.endcfl		 ==exsol.endcfl);
	assert(opts.tolerance	 ==exsol.tolerance);
	assert(opts.maxiter		 ==exsol.maxiter);
	assert(opts.firstinitcfl ==exsol.firstinitcfl);
	assert(opts.firstendcfl	 ==exsol.firstendcfl);
	assert(opts.firsttolerance ==exsol.firsttolerance);
	assert(opts.firstmaxiter ==exsol.firstmaxiter);
}

int main(int argc, char *argv[])
{
	int ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);

	po::options_description desc
		(std::string("FVENS test for option parsing. The first argument is the control info file.\n")
		 + "Further options");
	desc.add_options()("exact_solution_file", po::value<std::string>(),
	                   "Location of file containing the exact solution for this test");

	const po::variables_map cmdvars = parse_cmd_options(argc, argv, desc);

	if(cmdvars.count("help")) {
		std::cout << desc << std::endl;
		std::exit(0);
	}

	// Read control file
	const FlowParserOptions opts = parse_flow_controlfile(argc, argv, cmdvars);

	if(cmdvars.count("help")) {
		std::cout << desc << std::endl;
		std::exit(0);
	}

	std::ifstream solnfile(cmdvars["exact_solution_file"].as<std::string>());
	const FlowParserOptions exsol = parse_solution_file(solnfile);
	solnfile.close();

	compare(opts, exsol);

	ierr = PetscFinalize(); CHKERRQ(ierr);
	return ierr;
}
