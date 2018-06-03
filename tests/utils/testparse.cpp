#undef NDEBUG
#define DEBUG 1

#include <cassert>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include "utilities/aoptionparser.hpp"

using namespace std;
using namespace acfd;
namespace po = boost::program_options;

FlowParserOptions parse_solution_file(std::ifstream& inf)
{
	FlowParserOptions pd;
	a_real tempalpha;
	pd.bmarks.resize(2);
	inf >> pd.meshfile >> pd.vtu_output_file >> pd.logfile >> pd.lognres
	    >> pd.flowtype >> pd.gamma >> tempalpha >> pd.Minf >> pd.slipwall_marker >> pd.farfield_marker
	    >> pd.num_out_walls >> pd.surfnameprefix >> pd.vol_output_reqd
	    >> pd.sim_type >> pd.invflux >> pd.gradientmethod >> pd.limiter
	    >> pd.pseudotimetype >> pd.initcfl >> pd.endcfl >> pd.tolerance >> pd.maxiter
	    >> pd.firstinitcfl >> pd.firstendcfl >> pd.firsttolerance >> pd.firstmaxiter;
	pd.alpha = tempalpha * PI/180.0;
	return pd;
}

int compare(const FlowParserOptions& opts, const FlowParserOptions& exsol)
{
	assert(opts.meshfile == exsol.meshfile);
	assert(opts.vtu_output_file == exsol.vtu_output_file);
	assert(opts.logfile == exsol.logfile);
	assert(opts.lognres == exsol.lognres);
	assert(opts.flowtype == exsol.flowtype);
}

int main(int argc, char *argv[])
{
	po::options_description desc
		(std::string("FVENS test for option parsing. The first argument is the control info file.\n")
		 + "Further options");

	const po::variables_map cmdvars = parse_cmd_options(argc, argv, desc);

	const std::string exf = cmdvars["exact_solution_file"].as<std::string>();
	desc.add_options()("exact_solution_file", "Location of file containing the exact solution for \
this test");

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
	Parse_data exsol = parse_solution_file(solnfile);
	solnfile.close();
}
