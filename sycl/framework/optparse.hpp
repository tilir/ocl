//------------------------------------------------------------------------------
//
// Option parsing to use in testing system
//
//------------------------------------------------------------------------------
//
// This file is licensed after LGPL v3
// Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
//
//------------------------------------------------------------------------------

#pragma once

#include <cassert>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace std::string_literals;

class optparser_t final {
  po::options_description desc_ = "Available options"s;
  po::variables_map vm_;
  std::string platform_;
  std::string program_;
  bool quiet_ = false;
  bool parsed_ = false;

public:
  optparser_t() {
    desc_.add_options()("help", "Produce help message");
    desc_.add_options()("size", po::value<int>()->default_value(0),
                        "workload main size");
    desc_.add_options()("nreps", po::value<int>()->default_value(0),
                        "workload number of repetitions");
  }

  template <typename T>
  void add(std::string name, T defval, std::string description = "") {
    assert(!parsed_ && "Please do not add options after they are parsed");
    desc_.add_options()(name.c_str(), po::value<T>()->default_value(defval),
                        description.c_str());
  }

  template <typename T> T get(std::string name) const {
    assert(parsed_ && "Please do not query options before they are parsed");
    return vm_[name].as<T>();
  }

  void parse(int argc, char **argv) {
    assert(!parsed_ && "Please do not try to parse more then once");

    po::store(po::parse_command_line(argc, argv, desc_), vm_);
    po::notify(vm_);

    parsed_ = true;

    if (vm_.count("help") > 0) {
      std::cout << desc_ << std::endl;
      exit(0);
    }
  }

  bool parsed() const noexcept { return parsed_; }
};
