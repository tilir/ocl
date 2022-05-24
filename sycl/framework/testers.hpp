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

// sycl access mode synonyms
constexpr auto sycl_read = cl::sycl::access::mode::read;
constexpr auto sycl_write = cl::sycl::access::mode::write;
constexpr auto sycl_read_write = cl::sycl::access::mode::read_write;
constexpr auto sycl_atomic = cl::sycl::access::mode::atomic;

// local target and fence aliases
constexpr auto sycl_local = cl::sycl::access::target::local;
constexpr auto sycl_local_fence = cl::sycl::access::fence_space::local_space;
constexpr auto sycl_global_fence = cl::sycl::access::fence_space::global_space;

// kernel bundle type aliases
using IBundleTy = sycl::kernel_bundle<sycl::bundle_state::input>;
using OBundleTy = sycl::kernel_bundle<sycl::bundle_state::object>;
using EBundleTy = sycl::kernel_bundle<sycl::bundle_state::executable>;

// buffer property aliases
constexpr auto host_ptr = cl::sycl::property::buffer::use_host_ptr{};

// event and profiling aliases
constexpr auto EvtStart = sycl::info::event_profiling::command_start;
constexpr auto EvtEnd = sycl::info::event_profiling::command_end;
constexpr auto EvtStatus = sycl::info::event::command_execution_status;
constexpr auto EvtComplete = sycl::info::event_command_status::complete;

// milliseconds, microsecond and nanoseconds
static const double msec_per_sec = 1000.0;
static const double usec_per_sec = msec_per_sec * msec_per_sec;
static const double nsec_per_sec = msec_per_sec * msec_per_sec * msec_per_sec;

// global and local atomic references
template <typename T>
using global_atomic_ref = cl::sycl::ext::oneapi::atomic_ref<
    T, cl::sycl::memory_order::relaxed, cl::sycl::memory_scope::system,
    cl::sycl::access::address_space::global_space>;

template <typename T>
using local_atomic_ref = cl::sycl::ext::oneapi::atomic_ref<
    T, cl::sycl::memory_order::relaxed, cl::sycl::memory_scope::work_group,
    cl::sycl::access::address_space::local_space>;

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

struct NamedEvent {
  cl::sycl::event Evt_;
  std::string Name_;
  NamedEvent(cl::sycl::event Evt, std::string Name = "Unnamed")
      : Evt_(Evt), Name_(std::move(Name)) {}
};

using EvtVec_t = std::vector<NamedEvent>;
using EvtRet_t = std::optional<EvtVec_t>;

inline unsigned getTime(EvtRet_t Opt, bool Quiet = true) {
  auto AccTime = 0, EvtIdx = 0;
  if (!Opt.has_value())
    return AccTime;
  auto &&Evts = Opt.value();
  for (auto &&NEvt : Evts) {
    cl::sycl::event &Evt = NEvt.Evt_;
    if (!Quiet)
      std::cout << EvtIdx++ << " (" << NEvt.Name_ << "): ";
    auto Status = Evt.get_info<EvtStatus>();
    if (Status != EvtComplete) {
      if (!Quiet)
        std::cout << " [...] ";
      Evt.wait();
    }
    auto Start = Evt.get_profiling_info<EvtStart>();
    auto End = Evt.get_profiling_info<EvtEnd>();
    if (!Quiet)
      std::cout << ((End - Start) / nsec_per_sec) << std::endl;
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
