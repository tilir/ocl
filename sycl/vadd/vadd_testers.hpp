//------------------------------------------------------------------------------
//
// Generic code to test different variants of vector add
// Avoiding tons of boilerplate otherwise
//
// Macros to control things:
//  * inherited from testers.hpp: RUNHOST, MEASURE_NORMAL, INORD...
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

// problems with boost in OneAPI console on Windows
#ifdef USE_BOOST_OPTPARSE
#include "optparse.hpp"
#else
#include "optparse_alt.hpp"
#endif

#include "testers.hpp"

namespace sycltesters {

namespace vadd {

constexpr int BLOCK_SIZE = 256;
constexpr int LIST_SIZE = 1024;
constexpr int NREPS = 10;
constexpr int DEF_DETAILED = 0;
constexpr int DEF_QUIET = 0;

struct Config {
  bool Detailed, Quiet = false;
  int Bsz, Size, NReps;
};

inline Config read_config(int argc, char **argv) {
  Config Cfg;
  options::Parser OptParser;
  OptParser.template add<int>("bsz", BLOCK_SIZE, "size of block");
  OptParser.template add<int>("size", LIST_SIZE,
                              "size of vectors in bsz-units");
  OptParser.template add<int>("nreps", NREPS,
                              "number of repetitions in tester loop");
  OptParser.template add<int>("detailed", DEF_DETAILED, "detailed event view");
  OptParser.template add<int>("quiet", DEF_QUIET, "quiet mode for bulk runs");
  OptParser.parse(argc, argv);

  Cfg.Bsz = OptParser.template get<int>("size");
  Cfg.Size = OptParser.template get<int>("size") * Cfg.Bsz;
  Cfg.NReps = OptParser.template get<int>("nreps");
  Cfg.Detailed = OptParser.exists("detailed");
  if (OptParser.exists("quiet")) {
    Cfg.Quiet = true;
    qout.set(Cfg.Quiet);
  }
  return Cfg;
}

inline void dump_config_info(Config &Cfg) {
  qout << "Using vector size = " << Cfg.Size << "\n";
  qout << "Using #of repetitions = " << Cfg.NReps << "\n";
}

} // namespace vadd

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

template <typename T> struct VectorAddHost : public VectorAdd<T> {
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
  VectorAddTester(VectorAdd<T> &Vadder, int Sz, int Rep)
      : Vadder_(Vadder), Sz_(Sz), Rep_(Rep) {
    A_.resize(Sz_);
    B_.resize(Sz_);
    C_.resize(Sz_);
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
  std::pair<unsigned, unsigned> calculate() {
    // timer start
    unsigned EvtTiming = 0;
    qout << "Nreps = " << Rep_ << "\n";
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
  try {
    auto Cfg = vadd::read_config(argc, argv);
    qout << "Welcome to vector addition"
         << "\n";
    dump_config_info(Cfg);
    auto Q = set_queue();
    print_info(qout, Q.get_device());

#ifdef MEASURE_NORMAL
    VectorAddHost<int> VaddH{Q}; // Q unused for this derived class
    VectorAddTester<int> TesterH{VaddH, Cfg.Size, Cfg.NReps};
    TesterH.initialize();
    auto ElapsedH = TesterH.calculate();
    qout << "Measured host time: " << ElapsedH.first << "\n";
#endif

    VaddChildT Vadd{Q};
    VectorAddTester<typename VaddChildT::type> Tester{Vadd, Cfg.Size,
                                                      Cfg.NReps};

    qout << "Initializing"
         << "\n";
    Tester.initialize();

    qout << "Calculating"
         << "\n";
    auto Elapsed = Tester.calculate();

    qout << "Measured time: " << Elapsed.first / 1000.0 << "\n";

    auto ExecTime = Elapsed.second / nsec_per_sec;
    qout << "Pure execution time: " << ExecTime << "\n";

    // Quiet mode output: vector size, elapsed time
    if (Cfg.Quiet) {
      qout.set(!Cfg.Quiet);
      qout << Cfg.Size << " " << ExecTime << "\n";
      qout.set(Cfg.Quiet);
    }
  } catch (cl::sycl::exception const &err) {
    std::cerr << "SYCL ERROR: " << err.what() << "\n";
    abort();
  } catch (std::exception const &err) {
    std::cerr << "Exception: " << err.what() << "\n";
    abort();
  } catch (...) {
    std::cerr << "Unknown error\n";
    abort();
  }
  qout << "Everything is correct"
       << "\n";
}

} // namespace sycltesters
