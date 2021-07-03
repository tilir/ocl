//-----------------------------------------------------------------------------
//
// framecl platforms: platform and device information
//
//-----------------------------------------------------------------------------

#pragma once

#include <cassert>
#include <iomanip>
#include <iostream>
#include <vector>

#include "fchelper.hpp"

// clang-format off
#include "cldefs.h"
#include "CL/opencl.hpp"
// clang-format on

constexpr int field_name = 30;
constexpr int field_vendor = 23;
constexpr int field_version = 26;
constexpr int field_total = 80;

namespace framecl {

class platform_list_t {
  std::vector<cl::Platform> platforms_;

public:
  platform_list_t() {
    cl::Platform::get(&platforms_);
    assert(platforms_.size() > 0 && "No OpenCL platforms found");
  }

  cl::Platform select(std::string platform) const {
    for (auto p : platforms_) {
      if (std::string::npos != p.getInfo<CL_PLATFORM_NAME>().find(platform) ||
          std::string::npos != p.getInfo<CL_PLATFORM_VENDOR>().find(platform) ||
          std::string::npos !=
              p.getInfo<CL_PLATFORM_VERSION>().find(platform)) {
        return p;
      }
    }
    std::ostringstream os;
    os << "Error: can not select this platform\n";
    throw std::runtime_error(os.str());
  }

  void print_info(std::ostream &os) {
    os << std::setw(field_total) << std::setfill('*') << "*"
       << "\n";
    os << std::left << std::setfill(' ') << std::setw(field_name)
       << "* Platform name" << std::setw(field_vendor) << "* Vendor"
       << std::setw(field_version) << "* Version"
       << "*\n";
    os << std::setw(field_total) << std::setfill('*') << "*"
       << "\n";
    for (auto p : platforms_) {
      auto nm = do_trim("* " + p.getInfo<CL_PLATFORM_NAME>(), field_name);
      auto vd = do_trim("* " + p.getInfo<CL_PLATFORM_VENDOR>(), field_vendor);
      auto vr = do_trim("* " + p.getInfo<CL_PLATFORM_VERSION>(), field_version);
      os << std::left << std::setfill(' ') << std::setw(field_name) << nm
         << std::setw(field_vendor) << vd << std::setw(field_version) << vr
         << "*\n";
    }
    os << std::setw(field_total) << std::setfill('*') << "*"
       << "\n";
  }

  void print_devices(std::ostream &os, std::string platform) {
    auto p = select(platform);
    std::string pstr = "* Platform: " + p.getInfo<CL_PLATFORM_NAME>() + " ";

    cl::vector<cl::Device> devices;
    p.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    os << std::left << std::setw(field_total - 1) << std::setfill(' ') << pstr
       << "*\n";
    os << std::setw(field_total) << std::setfill('*') << "*"
       << "\n";
    os << std::left << std::setfill(' ') << std::setw(field_name)
       << "* Device name" << std::setw(field_vendor) << "* Vendor"
       << std::setw(field_version) << "* Version"
       << "*\n";
    os << std::setw(field_total) << std::setfill('*') << "*"
       << "\n";

    for (auto d : devices) {
      auto nm = do_trim("* " + d.getInfo<CL_DEVICE_NAME>(), field_name);
      auto vd = do_trim("* " + d.getInfo<CL_DEVICE_VENDOR>(), field_vendor);
      auto vr = do_trim("* " + d.getInfo<CL_DEVICE_VERSION>(), field_version);
      os << std::left << std::setfill(' ') << std::setw(field_name) << nm
         << std::setw(field_vendor) << vd << std::setw(field_version) << vr
         << "*\n";
    }
    os << std::setw(field_total) << std::setfill('*') << "*"
       << "\n";
  }

  void print_device_info(std::ostream &os, cl::Device d) {
    os << "* Device name: " << d.getInfo<CL_DEVICE_NAME>() << "\n";
    os << "* Device vendor: " << d.getInfo<CL_DEVICE_VENDOR>() << "\n";
    os << "* Device version: " << d.getInfo<CL_DEVICE_VERSION>() << "\n";
    os << "* Device type: " << d.getInfo<CL_DEVICE_TYPE>() << "\n";
    os << "* Device CU's: " << d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << "\n";
    os << "* Device global mem size: " << d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()
       << "\n";
    os << "* Device work group size: "
       << d.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << "\n";
    os << "* Device work item dims: "
       << d.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << "\n";
    os << "* Device work item sizes: "
       << d.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>() << "\n";
  }

  void print_detailed(std::ostream &os, std::string device) {
    for (auto p : platforms_) {
      cl::vector<cl::Device> devices;
      p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
      for (auto d : devices) {
        auto nm = d.getInfo<CL_DEVICE_NAME>();
        if (std::string::npos != nm.find(device))
          print_device_info(os, d);
      }
    }
  }

private:
  std::string do_trim(std::string s, size_t width) {
    if (s.size() >= width)
      s = s.substr(0, width - 4) + "...";
    return s;
  }
};

} // namespace framecl
