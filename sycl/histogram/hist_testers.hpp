//------------------------------------------------------------------------------
//
// Generic code to test different variants of histogram
// Avoiding tons of boilerplate otherwise
//
// Macros to control things:
//  * inherited from testers.hpp: RUNHOST, INORD...
//  -DMEASURE_NORMAL : measure with normal host code
//
// Options to control things:
// -bsz=<bsz> : block size
// -sz=<sz> : data size (in bsz-units)
// -hsz=<hsz> : number of buckets
// -gsz=<g> : global iteration space (in bsz-units)
// -lsz=<l> : local iteration space
// -vis=1 : visualize hist (use wisely) available only in measure_normal
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

#ifdef USE_BOOST_OPTPARSE
#include "optparse.hpp"
#else
#include "optparse_alt.hpp"
#endif

#include "testers.hpp"

// we will need global and local atomic references
template <typename T>
using local_atomic_ref = cl::sycl::ext::oneapi::atomic_ref<
    T, cl::sycl::memory_order::relaxed, cl::sycl::memory_scope::work_group,
    cl::sycl::access::address_space::local_space>;

template <typename T>
using global_atomic_ref = cl::sycl::ext::oneapi::atomic_ref<
    T, cl::sycl::memory_order::relaxed, cl::sycl::memory_scope::system,
    cl::sycl::access::address_space::global_space>;

namespace sycltesters {

constexpr int DEF_BSZ = 1024;
constexpr int DEF_SZ = 2000;
constexpr int DEF_HSZ = 32;
constexpr int DEF_GSZ = 10;
constexpr int DEF_LSZ = 8;
constexpr int DEF_VIS = 0;

struct Config {
  int Block, Sz, HistSz, GlobSz, LocSz, Vis;
};

template <typename T> class Histogramm {
  cl::sycl::queue DeviceQueue_;

public:
  using type = T;
  Histogramm(cl::sycl::queue &DeviceQueue) : DeviceQueue_(DeviceQueue) {}
  virtual EvtRet_t operator()(const T *Data, T *Bins, size_t NumData,
                              size_t NumBins) = 0;
  cl::sycl::queue &Queue() { return DeviceQueue_; }
  const cl::sycl::queue &Queue() const { return DeviceQueue_; }
  virtual ~Histogramm() {}
};

template <typename T> struct HistogrammHost : public Histogramm<T> {
  HistogrammHost(cl::sycl::queue &DeviceQueue) : Histogramm<T>(DeviceQueue) {}
  EvtRet_t operator()(const T *Data, T *Bins, size_t NumData,
                      size_t NumBins) override {
    for (int i = 0; i < NumData; i++) {
      assert(Data[i] < NumBins);
      Bins[Data[i]] += 1;
    }
    return {}; // nothing to construct as event
  }
};

template <typename T> class HistogrammTester {
  Histogramm<T> &Hist_;
  Timer Timer_;
  const T *Data_;
  size_t NumData_, NumBins_;
  std::vector<T> Bins_;

public:
  HistogrammTester(Histogramm<T> &Hist, const T *Data, size_t NumData,
                   size_t NumBins)
      : Hist_(Hist), Data_(Data), NumData_(NumData), NumBins_(NumBins),
        Bins_(NumBins) {}

  std::pair<unsigned, unsigned> calculate() {
    unsigned EvtTiming = 0;
    Timer_.start();
    EvtRet_t Ret = Hist_(Data_, Bins_.data(), NumData_, NumBins_);
    EvtTiming += getTime(Ret);
    Timer_.stop();
    return {Timer_.elapsed(), EvtTiming};
  }

  const T *getData() const { return Data_; }
  T *getref() { return Bins_.data(); }
};

template <typename T>
void dump_hist(std::ostream &Os, std::string Name, const T *Data, int Sz) {
  Os << Name << ":\n";
  std::ostream_iterator<T> OsIt(Os, " ");
  std::copy(Data, Data + Sz, OsIt);
  Os << "\n";
}

template <typename T>
void rand_initialize(T *Arr, size_t Sz, int Min, int Max) {
  Dice D(Min, Max);
  std::generate(Arr, Arr + Sz, [&] { return D(); });
}

inline Config read_config(int argc, char **argv) {
  Config Cfg;
  optparser_t OptParser;
  OptParser.template add<int>("bsz", DEF_BSZ, "block size");
  OptParser.template add<int>("sz", DEF_SZ,
                              "data size (in bsz-element blocks)");
  OptParser.template add<int>("hsz", DEF_HSZ, "number of buckets");
  OptParser.template add<int>("gsz", DEF_GSZ,
                              "global iteration space (in bsz-element blocks)");
  OptParser.template add<int>("lsz", DEF_LSZ, "local iteration space");
#ifdef MEASURE_NORMAL
  OptParser.template add<int>("vis", 0, "pass 1 to visualize matrices");
#endif
  OptParser.parse(argc, argv);

  Cfg.Block = OptParser.template get<int>("bsz");
  Cfg.Sz = OptParser.template get<int>("sz") * Cfg.Block;
  Cfg.HistSz = OptParser.template get<int>("hsz");
  Cfg.GlobSz = OptParser.template get<int>("gsz") * Cfg.Block;
  Cfg.LocSz = OptParser.template get<int>("lsz");
#ifdef MEASURE_NORMAL
  Cfg.Vis = OptParser.template get<int>("vis");
#endif

  if (Cfg.Block < 1 || Cfg.Sz < 1 || Cfg.HistSz < 1 || Cfg.GlobSz < 1 ||
      Cfg.LocSz < 1) {
    std::cerr << "Wrong parameters (expect all sizes >= 1)\n";
    std::terminate();
  }

  std::cout << "Block size: " << Cfg.Block << std::endl;
  std::cout << "Data size: " << Cfg.Sz << std::endl;
  std::cout << "Histogram size: " << Cfg.HistSz << std::endl;
  std::cout << "Global size: " << Cfg.GlobSz << std::endl;
  std::cout << "Local size: " << Cfg.LocSz << std::endl;
#ifdef MEASURE_NORMAL
  if (Cfg.Vis)
    std::cout << "Visual mode" << std::endl;
#endif
  return Cfg;
}

template <typename HistChildT> void test_sequence(int argc, char **argv) {
  std::cout << "Welcome to histogram" << std::endl;

  try {
    Config Cfg = read_config(argc, argv);
    auto Q = set_queue();
    print_info(std::cout, Q.get_device());

    std::cout << "Initializing" << std::endl;
    using Ty = typename HistChildT::type;
    std::vector<Ty> Data(Cfg.Sz);
    rand_initialize(Data.data(), Data.size(), 0, Cfg.HistSz - 1);

#ifdef MEASURE_NORMAL
    std::cout << "Calculating host" << std::endl;
    HistogrammHost<Ty> HistH{Q}; // Q unused for this derived class
    HistogrammTester<Ty> TesterH{HistH, Data.data(), Cfg.Sz, Cfg.HistSz};
    auto ElapsedH = TesterH.calculate();
    std::cout << "Measured host time: " << ElapsedH.first / msec_per_sec
              << std::endl;
#endif

    HistChildT Hist{Q, Cfg.GlobSz, Cfg.LocSz};

    HistogrammTester<Ty> Tester{Hist, Data.data(), Data.size(), Cfg.HistSz};

    std::cout << "Calculating gpu" << std::endl;
    auto Elapsed = Tester.calculate();

    std::cout << "Measured time: " << Elapsed.first / msec_per_sec << std::endl
              << "Pure execution time: " << Elapsed.second / nsec_per_sec
              << std::endl;

#ifdef MEASURE_NORMAL
    // verification with host result
    Ty *HostData = TesterH.getref();
    Ty *GPUData = Tester.getref();

    if (Cfg.Vis) {
      dump_hist(std::cout, "Data: ", Tester.getData(), Cfg.Sz);
      dump_hist(std::cout, "Host result", HostData, Cfg.HistSz);
      dump_hist(std::cout, "GPU result", GPUData, Cfg.HistSz);
    }

    for (int I = 0; I < Cfg.HistSz; ++I)
      if (HostData[I] != GPUData[I]) {
        std::cout << "Mismatch at: " << I << std::endl;
        std::cout << HostData[I] << " vs " << GPUData[I] << std::endl;
        return;
      }
#endif
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
  std::cout << "Everything is correct" << std::endl;
}

} // namespace sycltesters
