//------------------------------------------------------------------------------
//
// Option parsing to use in testing system
// Alternative version (without boost)
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

class optparser_t final {
  bool parsed_ = false;
  std::unordered_map<std::string, std::string> values;

public:
  optparser_t() {
    values["size"] = "0";
    values["nreps"] = "0";
  }

  template <typename T>
  void add(std::string name, T defval, std::string description = "") {
    std::ostringstream os;
    os << defval;
    values[name] = os.str();
  }

  template <typename T> T get(std::string name) const {
    if (!parsed_)
      throw std::runtime_error(
          "Please do not query options before they are parsed");
    if (!values.count(name))
      throw std::runtime_error("Option not present in list, use add");
    std::string val = values.find(name)->second;
    std::istringstream is{val};
    T ret;
    is >> ret;
    return ret;
  }

  void parse(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
      std::string_view optview = argv[i];
      auto trim_pos = optview.find_first_not_of('-');
      optview.remove_prefix(trim_pos);
      trim_pos = optview.find('=');
      if (trim_pos != optview.npos) {
        std::string_view valview = optview;
        optview.remove_suffix(optview.size() - trim_pos);
        valview.remove_prefix(trim_pos + 1);
        std::string val{valview};
        std::string opt{optview};
        if (!values.count(opt)) {
          std::cout << "option detected: " << opt << " : " << val << std::endl;
          throw std::runtime_error("Option not expected, use add");
        }
        values[opt] = val;
      }

      if (argv[i] == std::string("help")) {
        std::cout << "Available options:" << std::endl;
        for (auto &&x : values)
          std::cout << ("-" + x.first) << std::endl;
        exit(0);
      }
    }
    parsed_ = true;
  }

  bool parsed() const noexcept { return parsed_; }
};