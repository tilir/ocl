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

#include <CL/sycl.hpp>
#include <boost/optional.hpp>

#include <cassert>
#include <chrono>
#include <iostream>
#include <vector>

#include "optparse.hpp"

// convenient sycl mode synonyms
constexpr auto sycl_read = cl::sycl::access::mode::read;
constexpr auto sycl_write = cl::sycl::access::mode::write;
constexpr auto host_ptr = cl::sycl::property::buffer::use_host_ptr{};

// convenient namspaces
namespace esimd = sycl::ext::intel::experimental::esimd;

namespace sycltesters {

namespace chrono = std::chrono;

// defaults
constexpr int LIST_SIZE = 1024 * 1024 * 2;
constexpr int NREPS = 10;

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

using EvtRet_t = boost::optional<std::vector<cl::sycl::event>>;

template <typename T> class VectorAdd {
  cl::sycl::queue DeviceQueue_;

public:
  using type = T;
  VectorAdd(cl::sycl::queue &DeviceQueue) : DeviceQueue_(DeviceQueue) {}
  virtual EvtRet_t operator()(T const *AVec, T const *BVec, T *CVec,
                              size_t Sz) = 0;
  cl::sycl::queue &Queue() { return DeviceQueue_; }
  const cl::sycl::queue &Queue() const { return DeviceQueue_; }
  virtual ~VectorAdd() {}
};

template <typename T> class VectorAddHost : public VectorAdd<T> {
public:
  VectorAddHost(cl::sycl::queue &DeviceQueue) : VectorAdd<T>(DeviceQueue) {}
  EvtRet_t operator()(T const *AVec, T const *BVec, T *CVec,
                      size_t Sz) override {
    for (size_t I = 0; I < Sz; ++I)
      CVec[I] = AVec[I] + BVec[I];
    return {}; // nothing to construct as event
  }
};

template <typename T> class VectorAddTester {
  std::vector<T> A_, B_, C_;
  VectorAdd<T> &Vadder_;
  Timer Timer_;
  unsigned Sz_;
  unsigned Rep_;

public:
  VectorAddTester(VectorAdd<T> &Vadder, unsigned Sz, unsigned Rep)
      : Vadder_(Vadder), Sz_(Sz), Rep_(Rep) {
    A_.resize(Sz_);
    B_.resize(Sz_);
    C_.resize(Sz_);
  }

  void print_info(std::ostream &os) const {
    auto device = Vadder_.Queue().get_device();
    os << device.template get_info<cl::sycl::info::device::name>() << "\n";
    os << "Driver version: "
       << device.template get_info<cl::sycl::info::device::driver_version>()
       << "\n";
    os << device.template get_info<cl::sycl::info::device::opencl_c_version>()
       << "\n";
  }

  void initialize() {
    for (int i = 0; i < Sz_; i++) {
      A_[i] = i;
      B_[i] = Sz_ - i;
      C_[i] = 0;
    }
  }

  unsigned getTime(EvtRet_t Opt) {
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

  // to have perf measurements we are doing in loop:
  // C = A + B;
  // A = B + C;
  // B = C + A;
  std::pair<unsigned, unsigned> calculate() {
    // timer start
    unsigned EvtTiming = 0;
    std::cout << "Nreps = " << Rep_ << std::endl;
    Timer_.start();
    // loop
    for (int i = 0; i < Rep_; ++i) {
      EvtRet_t Ret;
      Ret = Vadder_(A_.data(), B_.data(), C_.data(), Sz_);
      EvtTiming += getTime(Ret);
      Ret = Vadder_(B_.data(), C_.data(), A_.data(), Sz_);
      EvtTiming += getTime(Ret);
      Ret = Vadder_(C_.data(), A_.data(), B_.data(), Sz_);
      EvtTiming += getTime(Ret);
    }
    Timer_.stop();
    return {Timer_.elapsed(), EvtTiming};
  }
};

template <typename VaddChildT> void test_sequence(int argc, char **argv) {
  std::cout << "Welcome to vector addition" << std::endl;

  try {
    optparser_t OptParser;
    OptParser.parse(argc, argv);

    unsigned Size = OptParser.template get<int>("size");
    unsigned NReps = OptParser.template get<int>("nreps");

    if (Size == 0)
      Size = LIST_SIZE;
    if (NReps == 0)
      NReps = NREPS;

    std::cout << "Using vector size = " << Size << std::endl;
    std::cout << "Using #of repetitions = " << NReps << std::endl;

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

#ifdef MEASURE_NORMAL
    VectorAddHost<int> VaddH{Q}; // Q unused for this derived class
    VectorAddTester<int> TesterH{VaddH, Size, NReps};
    TesterH.initialize();
    auto ElapsedH = TesterH.calculate();
    std::cout << "Measured host time: " << ElapsedH.first << std::endl;
#endif

    VaddChildT Vadd{Q};
    VectorAddTester<typename VaddChildT::type> Tester{Vadd, Size, NReps};

    Tester.print_info(std::cout);

    std::cout << "Initializing" << std::endl;
    Tester.initialize();

    std::cout << "Calculating" << std::endl;
    auto Elapsed = Tester.calculate();

    std::cout << "Measured time: " << Elapsed.first / 1000.0 << std::endl;
    std::cout << "Pure execution time: " << Elapsed.second / 1000000000.0
              << std::endl;
  } catch (cl::sycl::exception const &err) {
    std::cerr << "SYCL ERROR: " << err.what() << ":\n";
    abort();
  }
  std::cout << "Everything is correct" << std::endl;
}

} // namespace sycltesters
