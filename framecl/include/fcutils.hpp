//-----------------------------------------------------------------------------
//
// framecl utilities: options, timer, etc
//
//-----------------------------------------------------------------------------

#pragma once

#include <cassert>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>

#include "fcplatform.hpp"

namespace po = boost::program_options;
using namespace std::string_literals;

namespace framecl {

class optparser_t final {
  po::options_description desc_ = "Available options"s;
  po::variables_map vm_;
  bool quiet_ = false;
  std::string platform_;
  std::string program_;
  bool parsed_ = false;

public:
  optparser_t() {
    desc_.add_options()("help", "Produce help message");
    desc_.add_options()("list", "List available platforms");
    desc_.add_options()(
        "list-devices", po::value<std::string>(),
        "List available devices for given platform (matches substr in name)");
    desc_.add_options()(
        "info", po::value<std::string>(),
        "Detailed info about given device (matches substr in name)");
    desc_.add_options()("platform", po::value<std::string>(),
                        "Select platform (matches substr in name or vendor)");
    desc_.add_options()(
        "device", po::value<std::string>(),
        "Limit devices in context to given (matches substr in name)");
    desc_.add_options()("program", po::value<std::string>(),
                        "Select file with program");
    desc_.add_options()("quiet", po::bool_switch()->default_value(false),
                        "Suppress almost all messages");
    desc_.add_options()("verbose", po::bool_switch()->default_value(false),
                        "Show additional debug messages");
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

    if (vm_.count("list") > 0) {
      platform_list_t plist;
      plist.print_info(std::cout);
      exit(0);
    }

    if (vm_.count("list-devices") > 0) {
      auto ldval = vm_["list-devices"].as<std::string>();
      platform_list_t plist;
      plist.print_devices(std::cout, ldval);
      exit(0);
    }

    if (vm_.count("info") > 0) {
      auto infoval = vm_["info"].as<std::string>();
      platform_list_t plist;
      plist.print_detailed(std::cout, infoval);
      exit(0);
    }

    if (vm_.count("platform") > 0)
      platform_ = vm_["platform"].as<std::string>();

    if (vm_.count("program") > 0)
      program_ = vm_["program"].as<std::string>();

    quiet_ = vm_["quiet"].as<bool>();
    if (!quiet_) {
      std::cout << "Info: targeting OCL_VERSION = " << OCL_VERSION
                << " rebuild to change" << std::endl;
      std::cout << "Info: run with --help for option list" << std::endl;
    }
  }

  template <typename T, typename F>
  int check(std::string name, F checker, std::string emsg = "") const {
    assert(parsed_ &&
           "Please do not invoke checkers before options are parsed");
    T val = vm_[name].as<T>();
    if (!checker(val)) {
      std::ostringstream os;
      os << "incorrect value for option " << name;
      if (emsg != "")
        os << ", " << emsg;
      os << std::endl;
      throw std::runtime_error(os.str());
    }

    return val;
  }

  bool quiet() const noexcept {
    assert(parsed_ && "Please do not query options before they are parsed");
    return quiet_;
  }

  bool verbose() const noexcept {
    assert(parsed_ && "Please do not query options before they are parsed");
    return vm_["verbose"].as<bool>();
  }

  std::string platform() const {
    assert(parsed_ && "Please do not query options before they are parsed");
    return platform_;
  }

  void require(std::string what, std::string adderr) const {
    assert(parsed_ &&
           "Please do not check options for existence before they are parsed");
    if (0 == vm_.count(what)) {
      std::ostringstream os;
      os << "option " << what << " not present\n";
      os << adderr << "\n";
      throw std::runtime_error(os.str());
    }
  }

  void require_platform(std::string adderr) const {
    require("platform", adderr);
  }

  std::string program() const {
    assert(parsed_ && "Please do not query options before they are parsed");
    return program_;
  }

  void require_program(std::string adderr) const { require("program", adderr); }

  bool parsed() const noexcept { return parsed_; }
};

}; // namespace framecl