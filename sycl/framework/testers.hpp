//------------------------------------------------------------------------------
//
// Generic code to test different variants of sycl programs
// Avoiding tons of boilerplate otherwise
//
// Macros to control things:
//  -DINORD          : use inorder queues
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
#include <random>
#include <vector>

#include <CL/sycl.hpp>

// convenient sycl access mode synonyms
constexpr auto sycl_read = cl::sycl::access::mode::read;
constexpr auto sycl_write = cl::sycl::access::mode::write;
constexpr auto sycl_read_write = cl::sycl::access::mode::read_write;
constexpr auto sycl_atomic = cl::sycl::access::mode::atomic;

// local target and fence aliases
constexpr auto sycl_local = cl::sycl::access::target::local;
constexpr auto sycl_local_fence = cl::sycl::access::fence_space::local_space;
constexpr auto sycl_global_fence = cl::sycl::access::fence_space::global_space;

// convenient buffer property aliases
constexpr auto host_ptr = cl::sycl::property::buffer::use_host_ptr{};

// milliseconds, microsecond and nanoseconds
static const double msec_per_sec = 1000.0;
static const double usec_per_sec = msec_per_sec * msec_per_sec;
static const double nsec_per_sec = msec_per_sec * msec_per_sec * msec_per_sec;

// convenient namspaces
namespace esimd = sycl::ext::intel::experimental::esimd;
namespace chrono = std::chrono;

namespace sycltesters {

class Timer {
  chrono::high_resolution_clock::time_point Start, Fin;
  bool Started = false;

public:
  Timer() = default;
  void start() {
    assert(!Started);
    Started = true;
    Start = chrono::high_resolution_clock::now();
  }
  void stop() {
    assert(Started);
    Started = false;
    Fin = chrono::high_resolution_clock::now();
  }
  unsigned elapsed() {
    assert(!Started);
    auto Elps = Fin - Start;
    auto Msec = chrono::duration_cast<chrono::milliseconds>(Elps);
    return Msec.count();
  }
};

inline std::ostream &print_info(std::ostream &Os, cl::sycl::device D) {
  Os << D.template get_info<cl::sycl::info::device::name>() << "\n";
  Os << "Driver version: "
     << D.template get_info<cl::sycl::info::device::driver_version>() << "\n";
  Os << D.template get_info<cl::sycl::info::device::opencl_c_version>() << "\n";
  return Os;
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

  // use env "SYCL_DEVICE_FILTER=cpu" to run on host
  cl::sycl::default_selector Sel;
  cl::sycl::queue Q{Sel, Exception_handler, PropList};
  return Q;
}

struct Dice {
  std::uniform_int_distribution<int> Uid;

  Dice(int Min, int Max) : Uid(Min, Max) {}
  int operator()() const {
    static std::random_device Rd;
    static std::mt19937 Rng{Rd()};
    return Uid(Rng);
  }
};

template <typename It>
void rand_initialize(It Begin, It End, int Min, int Max) {
  Dice D(Min, Max);
  std::generate(Begin, End, [&] { return D(); });
}

template <typename It, typename Os>
void visualize_seq(It Begin, It End, Os &Stream) {
  using Ty = typename std::iterator_traits<It>::value_type;
  std::ostream_iterator<Ty> Out{Stream, " "};
  std::copy(Begin, End, Out);
  Stream << "\n";
}

} // namespace sycltesters
