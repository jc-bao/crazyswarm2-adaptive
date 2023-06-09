#include <iostream>

#include <boost/program_options.hpp>
#include <crazyflie_cpp/Crazyflie.h>
#include "logger.hpp"

int main(int argc, char **argv)
{

  std::string uri;
  std::string defaultUri("radio://0/80/2M/E7E7E7E7E7");
  bool verbose = false;

  namespace po = boost::program_options;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("uri", po::value<std::string>(&uri)->default_value(defaultUri), "unique ressource identifier")
    ("verbose,v", "verbose output")
  ;

  try
  {
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 0;
    }
    verbose = vm.count("verbose");
  }
  catch(po::error& e)
  {
    std::cerr << e.what() << std::endl << std::endl;
    std::cerr << desc << std::endl;
    return 1;
  }

  try
  {
    CrazyflieToolsLogger logger(verbose);
    Crazyflie cf(uri, logger);
    cf.requestParamToc();
    
    std::for_each(cf.paramsBegin(), cf.paramsEnd(),
      [&cf](const Crazyflie::ParamTocEntry& entry)
      {
        std::cout << entry.group << "." << entry.name << " (";
        switch (entry.type) {
        case Crazyflie::ParamTypeUint8:
          std::cout << "uint8";
          break;
        case Crazyflie::ParamTypeInt8:
          std::cout << "int8";
          break;
        case Crazyflie::ParamTypeUint16:
          std::cout << "uint16";
          break;
        case Crazyflie::ParamTypeInt16:
          std::cout << "int16";
          break;
        case Crazyflie::ParamTypeUint32:
          std::cout << "uint32";
          break;
        case Crazyflie::ParamTypeInt32:
          std::cout << "int32";
          break;
        case Crazyflie::ParamTypeFloat:
          std::cout << "float";
          break;
        }
        if (entry.readonly) {
          std::cout << ", readonly";
        }
        std::cout << ") value: ";

        switch (entry.type) {
        case Crazyflie::ParamTypeUint8:
          std::cout << (int)cf.getParam<uint8_t>(entry.id);
          break;
        case Crazyflie::ParamTypeInt8:
          std::cout << (int)cf.getParam<int8_t>(entry.id);
          break;
        case Crazyflie::ParamTypeUint16:
          std::cout << cf.getParam<uint16_t>(entry.id);
          break;
        case Crazyflie::ParamTypeInt16:
          std::cout << cf.getParam<int16_t>(entry.id);
          break;
        case Crazyflie::ParamTypeUint32:
          std::cout << cf.getParam<uint32_t>(entry.id);
          break;
        case Crazyflie::ParamTypeInt32:
          std::cout << cf.getParam<int32_t>(entry.id);
          break;
        case Crazyflie::ParamTypeFloat:
          std::cout << cf.getParam<float>(entry.id);
          break;
        }

        std::cout << std::endl;
      }
    );

    return 0;
  }
  catch(std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
