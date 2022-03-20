//------------------------------------------------------------------------------
//
// Generic code to test different variants of sycl programs
// Avoiding tons of boilerplate otherwise
//
// Macros to control things:
//  -DRUNHOST        : run as a host code (debugging, etc)
//  -DINORD          : use inorder queues
//  -DMEASURE_NORMAL : measure with normal host code
//
//------------------------------------------------------------------------------
//
// This file is licensed after LGPL v3
// Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
//
//------------------------------------------------------------------------------

#pragma once

#include <cassert>
#include <chrono>
#include <iostream>
#include <vector>

#include <CL/sycl.hpp>

// convenient sycl mode synonyms
constexpr auto sycl_read = cl::sycl::access::mode::read;
constexpr auto sycl_write = cl::sycl::access::mode::write;
constexpr auto host_ptr = cl::sycl::property::buffer::use_host_ptr{};

// convenient namspaces
namespace esimd = sycl::ext::intel::experimental::esimd;
namespace chrono = std::chrono;

namespace sycltesters {

class Timer {
  chrono::high_resolution_clock::time_point start_, fin_;
  bool started_ = false;

public:
  Timer() = default;
  void start() {
    assert(!started_);
    started_ = true;
    start_ = chrono::high_resolution_clock::now();
  }
  void stop() {
    assert(started_);
    started_ = false;
    fin_ = chrono::high_resolution_clock::now();
  }
  unsigned elapsed() {
    assert(!started_);
    auto elps = fin_ - start_;
    auto msec = chrono::duration_cast<chrono::milliseconds>(elps);
    return msec.count();
  }
};

inline std::ostream &print_info(std::ostream &os, cl::sycl::device d) {
  os << d.template get_info<cl::sycl::info::device::name>() << "\n";
  os << "Driver version: "
     << d.template get_info<cl::sycl::info::device::driver_version>() << "\n";
  os << d.template get_info<cl::sycl::info::device::opencl_c_version>() << "\n";
  return os;
}

using EvtRet_t = std::optional<std::vector<cl::sycl::event>>;

inline unsigned getTime(EvtRet_t Opt) {
  auto AccTime = 0;
  if (!Opt.has_value())
    return AccTime;
  auto &&Evts = Opt.value();
  for (auto &&Evt : Evts) {
    auto Start =
        Evt.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto End =
        Evt.get_profiling_info<sycl::info::event_profiling::command_end>();
    AccTime += End - Start;
  }
  return AccTime;
}

} // namespace sycltesters
