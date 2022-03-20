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
#include <random>
#include <vector>

#include <CL/sycl.hpp>

// convenient sycl mode synonyms
constexpr auto sycl_read = cl::sycl::access::mode::read;
constexpr auto sycl_write = cl::sycl::access::mode::write;
constexpr auto sycl_read_write = cl::sycl::access::mode::read_write;
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

inline cl::sycl::queue set_queue() {
  auto Exception_handler = [](sycl::exception_list e_list) {
    for (std::exception_ptr const &e : e_list)
      std::rethrow_exception(e);
  };

#ifdef INORD
  cl::sycl::property_list PropList{
      sycl::property::queue::in_order(),
      cl::sycl::property::queue::enable_profiling()};
#else
  cl::sycl::property_list PropList{
      cl::sycl::property::queue::enable_profiling()};
#endif

#ifdef RUNHOST
  cl::sycl::host_selector Hsel;
  cl::sycl::queue Q{Hsel, Exception_handler, PropList};
#else
  cl::sycl::gpu_selector GPsel;
  cl::sycl::queue Q{GPsel, Exception_handler, PropList};
#endif

  return Q;
}

struct Dice {
  std::uniform_int_distribution<int> uid;

  Dice(int min, int max) : uid(min, max) {}
  int operator()() {
    static std::random_device rd;
    static std::mt19937 rng{rd()};
    return uid(rng);
  }
};

} // namespace sycltesters
