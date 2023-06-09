#include <iostream>
#include <thread>

#include <boost/program_options.hpp>
#include <crazyflie_cpp/Crazyflie.h>
#include "logger.hpp"

void onLogCustom(uint32_t time_in_ms, const std::vector<float>* values, void* /*userData*/)
{
  std::cout << time_in_ms;
  for (const float v : *values) {
    std::cout << "," << v;
  }
  std::cout << std::endl;
}

int main(int argc, char **argv)
{

  std::string uri;
  std::string defaultUri("radio://0/80/2M/E7E7E7E7E7");
  std::vector<std::string> vars;
  uint32_t period;
  bool verbose = false;

  namespace po = boost::program_options;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("var", po::value<std::vector<std::string>>(&vars)->multitoken(), "variable names to log")
    ("period", po::value<uint32_t>(&period)->default_value(10), "sampling period in ms")
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
    cf.requestLogToc();

    std::function<void(uint32_t, const std::vector<float>*, void*)> cb = std::bind(&onLogCustom, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    LogBlockGeneric logBlock(&cf, vars, nullptr, cb);
    logBlock.start(period / 10);

    while (true) {
      cf.sendPing();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  catch(std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
