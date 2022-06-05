//------------------------------------------------------------------------------
//
// Generic code to test different variants of sycl programs
// Avoiding tons of boilerplate otherwise
//
// Macros to control things: see comment in CMakeLists.txt
//
//------------------------------------------------------------------------------
//
// This file is licensed after LGPL v3
// Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
//
//------------------------------------------------------------------------------

#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <iterator>
#include <vector>

#include <CL/sycl.hpp>

#include "dice.hpp"
#include "qstream.hpp"
#include "simplemath.hpp"
#include "syclconst.hpp"
#include "timers.hpp"

namespace sycltesters {

template <typename OsTy> OsTy &print_info(OsTy &Os, sycl::device D) {
  auto Name = D.template get_info<info::device::name>();
  auto Version = D.template get_info<info::device::version>();
  auto Vendor = D.template get_info<info::device::vendor>();
  auto DriverVersion = D.template get_info<info::device::driver_version>();
  auto OCLVersion = D.template get_info<info::device::opencl_c_version>();
  Os << "Name: " << Name << "\n";
  Os << "Version: " << Version << "\n";
  Os << "Vendor: " << Vendor << "\n";
  Os << "Driver: " << DriverVersion << "\n";
  Os << "OpenCL: " << OCLVersion << "\n";
  return Os;
}

inline sycl::queue set_queue() {
  auto Exception_handler = [](sycl::exception_list e_list) {
    for (std::exception_ptr const &e : e_list)
      std::rethrow_exception(e);
  };

#ifdef INORD
  sycl::property_list PropList{sycl::property::queue::in_order(),
                               sycl::property::queue::enable_profiling()};
#else
  sycl::property_list PropList{sycl::property::queue::enable_profiling()};
#endif

  // use env "SYCL_DEVICE_FILTER=cpu" to run on host
  sycl::default_selector Sel;
  sycl::queue Q{Sel, Exception_handler, PropList};
  return Q;
}

template <typename It, typename Os>
void visualize_seq(It Begin, It End, Os &Stream) {
  using Ty = typename std::iterator_traits<It>::value_type;
  std::ostream_iterator<Ty> Out{Stream, " "};
  std::copy(Begin, End, Out);
  Stream << "\n";
}

} // namespace sycltesters
