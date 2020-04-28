
#include <string>
#include <iostream>

#include <boltview/host_image.h>
#include <boltview/create_view.h>
#include <boltview/for_each.h>
#include <boltview/math/vector.h>
#include "io.h"
#include "cli.h"

#include <boost/program_options.hpp>

namespace po = boost::program_options;


void runOnDevice(bolt::Int2 size, bolt::Region<2, float> domain);
void runOnHost(bolt::Int2 size, bolt::Region<2, float> domain);



int main(int argc, char** argv) {
	try {
		bolt::Int2 output_size;
		bolt::Float2 domain_corner;
		bolt::Float2 domain_extents;
		po::options_description desc(
				"Generate image of the Mandelbrot's set.\n"
				"Example: './fractal -c \"[-1.03, 0.29]\" -e \"[0.06, 0.045]\" --device'\n\n"
				"Allowed options");
		desc.add_options()
			("help,h", "produce help message")
			("host", "execute on host")
			("device", "execute on device")
			("size,s", po::value<bolt::Int2>(&output_size)->default_value(bolt::Int2(4282, 2848)), "Size of the output image.")
			("domain_corner,c", po::value<bolt::Float2>(&domain_corner)->default_value(bolt::Float2(-2.5f, -1.0)), "Corner coordinates of the domain over which we will be computing.")
			("domain_extents,e", po::value<bolt::Float2>(&domain_extents)->default_value(bolt::Float2(3.5f, 2.0f)), "Size of the domain over which we will be computing.")
			;

		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);

		if (vm.count("help")) {
		    std::cout << desc << "\n";
		    return 1;
		}

		/* { Float2(-1.03f, 0.29f), Float2(0.06f, 0.045f) }*/
		bolt::Region<2, float> domain = { domain_corner, domain_extents };
		if (vm.count("host") > 0 || vm.count("device") == 0) {
			std::cout << "Generating the image on host...\n";
			runOnHost(output_size, domain);
		}
		if (vm.count("device") > 0) {
			std::cout << "Generating the image on device...\n";
			runOnDevice(output_size, domain);
		}

	} catch (std::exception &e) {
		std::cout << "boost::diagnostic_information(e):\n" << boost::diagnostic_information(e) << std::endl;
		return 1;
	}

	return 0;

}
