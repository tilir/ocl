//------------------------------------------------------------------------------
//
// Generic code to test different variants of sycl programs
// Avoiding tons of boilerplate otherwise
//
// Macros to control things:
//  -DRUNCPU         : run on CPU
//  -DMEASURE_NORMAL : measure with normal host code
//
//------------------------------------------------------------------------------
//
// This file is licensed after LGPL v3
// Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
//
//------------------------------------------------------------------------------

#include <CL/sycl.hpp>

#include <cassert>
#include <chrono>
#include <iostream>
#include <vector>

// convenient sycl mode synonyms
constexpr auto sycl_read = cl::sycl::access::mode::read;
constexpr auto sycl_write = cl::sycl::access::mode::write;

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

template <typename T>
class VectorAdd {
  cl::sycl::queue DeviceQueue_;
public:
  using type = T;
  VectorAdd(cl::sycl::queue &DeviceQueue) : DeviceQueue_(DeviceQueue) {}
  virtual void operator()(T const *AVec, T const *BVec, T *CVec, size_t Sz) = 0;
  cl::sycl::queue &Queue() { return DeviceQueue_; }
  const cl::sycl::queue &Queue() const { return DeviceQueue_; }
  virtual ~VectorAdd() {}  
};

template <typename T>
class VectorAddHost : public VectorAdd<T> {
public:
  VectorAddHost(cl::sycl::queue &DeviceQueue) : VectorAdd<T>(DeviceQueue) {}
  void operator()(T const *AVec, T const *BVec, T *CVec, size_t Sz) override {
    for (size_t I = 0; I < Sz; ++I)
      CVec[I] = AVec[I] + BVec[I];
  }
};
 
template <typename T>
class VectorAddTester {
  std::vector<T> A_, B_, C_;
  VectorAdd<T> &Vadder_;
  Timer Timer_;
  unsigned Sz_;
  unsigned Rep_;
public:
  VectorAddTester(VectorAdd<T> &Vadder, unsigned Sz = LIST_SIZE, unsigned Rep = NREPS) : Vadder_(Vadder), Sz_(Sz), Rep_(Rep) {
    A_.resize(Sz_);
    B_.resize(Sz_);
    C_.resize(Sz_);
  }
  
  void print_info(std::ostream &os) const {
    auto device = Vadder_.Queue().get_device();
    os << device.template get_info<cl::sycl::info::device::name>() << "\n";
    os << "Driver version: "
       << device.template get_info<cl::sycl::info::device::driver_version>() << "\n";
    os << device.template get_info<cl::sycl::info::device::opencl_c_version>() << "\n";
  }

  void initialize() {
    for (int i = 0; i < Sz_; i++) {
      A_[i] = i;
      B_[i] = Sz_ - i;
      C_[i] = 0;
    }
  }

  // to have perf measurements we are doing in loop:
  // C = A + B;
  // A = B + C;
  // B = C + A;
  unsigned calculate() {
    // timer start
    Timer_.start();
    // loop
    for (int i = 0; i < Rep_; ++i) {
      Vadder_(A_.data(), B_.data(), C_.data(), LIST_SIZE);
      Vadder_(B_.data(), C_.data(), A_.data(), LIST_SIZE);
      Vadder_(C_.data(), A_.data(), B_.data(), LIST_SIZE);
    }
    Timer_.stop();
    return Timer_.elapsed();
  }
};

template <typename VaddChildT>
void test_sequence(int argc, char **argv) {
  std::cout << "Welcome to vector addition" << std::endl;

  try {
#ifdef RUNCPU
    cl::sycl::cpu_selector CPsel;
    cl::sycl::queue Q{CPsel};
#else
    cl::sycl::gpu_selector GPsel;
    cl::sycl::queue Q{GPsel};
#endif

#ifdef MEASURE_NORMAL
    VectorAddHost<int> VaddH{Q}; // Q unused for this derived class
    VectorAddTester<int> TesterH{VaddH};
    TesterH.initialize();
    unsigned ElapsedH = TesterH.calculate();
    std::cout << "Measured host time: " << ElapsedH << std::endl;
#endif

    VaddChildT Vadd{Q};
    VectorAddTester<typename VaddChildT::type> Tester{Vadd};

    Tester.print_info(std::cout);

    std::cout << "Initializing" << std::endl;
    Tester.initialize();

    std::cout << "Calculating" << std::endl;
    auto Elapsed = Tester.calculate();

    std::cout << "Measured time: " << Elapsed << std::endl;
  } catch (cl::sycl::exception const &err) {
    std::cerr << "SYCL ERROR: " << err.what() << ":\n";
    abort();
  }
  std::cout << "Everything is correct" << std::endl;
}

}

