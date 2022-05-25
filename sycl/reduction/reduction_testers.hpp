//------------------------------------------------------------------------------
//
// Generic code to test different variants of reduction
// Avoiding tons of boilerplate otherwise
//
// Macros to control things:
//  * inherited from testers.hpp: RUNHOST, INORD...
//  -DMEASURE_NORMAL : measure with normal host code
//
// Options to control things:
// -bsz=<bsz> : block size
// -sz=<sz> : data size (in bsz-units)
// -gsz=<g> : global iteration space (in bsz-units)
// -lsz=<l> : local iteration space
// -val=<val> : fill data with val for debug
// -detailed=1 : detailed report from event
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
#include <numeric>
#include <vector>

#include <CL/sycl.hpp>

#ifdef USE_BOOST_OPTPARSE
#include "optparse.hpp"
#else
#include "optparse_alt.hpp"
#endif

#include "testers.hpp"

namespace sycltesters {

// we have sycltesters -> Config in multiple headers,
// so this namespace is required in each.
namespace reduce {

constexpr int DEF_BSZ = 128;
constexpr int DEF_SZ = 1024; // data size in blocks
constexpr int DEF_GSZ = 64;  // global iteration space in blocks
constexpr int DEF_LSZ = 32;
constexpr int DEF_VAL = 0;
constexpr int DEF_DETAILED = 0;

struct Config {
  bool ValExists, Detailed;
  int Block, Sz, GlobSz, LocSz, Val;
};

inline Config read_config(int argc, char **argv) {
  Config Cfg;
  options::Parser OptParser;
  OptParser.template add<int>("bsz", DEF_BSZ, "block size");
  OptParser.template add<int>("sz", DEF_SZ,
                              "data size (in bsz-element blocks)");
  OptParser.template add<int>("gsz", DEF_GSZ,
                              "global iteration space (in bsz-element blocks)");
  OptParser.template add<int>("lsz", DEF_LSZ, "local iteration space");
  OptParser.template add<int>("val", DEF_VAL, "fill data with given value");
  OptParser.template add<int>("detailed", DEF_DETAILED, "detailed event view");
  OptParser.parse(argc, argv);

  Cfg.Block = OptParser.template get<int>("bsz");
  Cfg.Sz = OptParser.template get<int>("sz") * Cfg.Block;
  Cfg.GlobSz = OptParser.template get<int>("gsz") * Cfg.Block;
  Cfg.LocSz = OptParser.template get<int>("lsz");
  Cfg.ValExists = OptParser.exists("val");
  Cfg.Val = OptParser.template get<int>("val");
  Cfg.Detailed = OptParser.exists("detailed");

  if (Cfg.Block < 1 || Cfg.Sz < 1 || Cfg.GlobSz < 1 || Cfg.LocSz < 1) {
    std::cerr << "Wrong parameters (expect all sizes >= 1)\n";
    std::terminate();
  }

  std::cout << "Block size: " << Cfg.Block << std::endl;
  std::cout << "Data size: " << Cfg.Sz << std::endl;
  std::cout << "Global size: " << Cfg.GlobSz << std::endl;
  std::cout << "Local size: " << Cfg.LocSz << std::endl;
  if (Cfg.ValExists)
    std::cout << "Filling with" << Cfg.Val << std::endl;
  if (Cfg.Detailed)
    std::cout << "Detailed events" << std::endl;
  std::cout.flush();
  return Cfg;
}
} // namespace reduce

template <typename T> class Reduction {
  sycl::queue DeviceQueue_;
  EBundleTy ExeBundle_;

public:
  using type = T;
  Reduction(sycl::queue &DeviceQueue, EBundleTy ExeBundle)
      : DeviceQueue_(DeviceQueue), ExeBundle_(ExeBundle) {}
  virtual EvtRet_t operator()(const T *Data, size_t NumData, T &Result) = 0;
  sycl::queue &Queue() { return DeviceQueue_; }
  EBundleTy Bundle() const { return ExeBundle_; }
  const sycl::queue &Queue() const { return DeviceQueue_; }
  virtual ~Reduction() {}
};

template <typename T> struct ReductionHost : public Reduction<T> {
  ReductionHost(sycl::queue &DeviceQueue, EBundleTy ExeBundle)
      : Reduction<T>(DeviceQueue, ExeBundle) {}
  EvtRet_t operator()(const T *Data, size_t NumData, T &Result) override {
    Result = std::reduce(Data, Data + NumData);
    return {}; // nothing to construct as event
  }
};

template <typename T> class ReductionTester {
  Reduction<T> &Reduce_;
  const T *Data_;
  reduce::Config Cfg_;

public:
  ReductionTester(Reduction<T> &Reduce, const T *Data, reduce::Config Cfg)
      : Reduce_(Reduce), Data_(Data), Cfg_(Cfg) {}

  std::pair<unsigned, unsigned> calculate(T &Result) {
    Timer Tm;
    Tm.start();
    EvtRet_t Ret = Reduce_(Data_, Cfg_.Sz, Result);
    Tm.stop();
    auto EvtTiming = getTime(Ret, Cfg_.Detailed ? false : true);
    return {Tm.elapsed(), EvtTiming};
  }

  const T *data() const { return Data_; }
};

template <typename ReductionChildT, typename Ty>
ReductionTester<Ty> single_reduce_sequence(sycl::queue &Q, reduce::Config Cfg,
                                           Ty *Data, EBundleTy ExeBundle) {
#if defined(MEASURE_NORMAL)
  std::cout << "Calculating host" << std::endl;
  ReductionHost<Ty> ReductionHost{Q, ExeBundle}; // both args here unused
  ReductionTester<Ty> TesterH{ReductionHost, Data, Cfg};
  Ty ResultH;
  auto ElapsedH = TesterH.calculate(ResultH);
  std::cout << "Measured host time: " << ElapsedH.first / msec_per_sec
            << std::endl;
#endif

  ReductionChildT Reduce{Q, ExeBundle, Cfg};

  ReductionTester<Ty> Tester{Reduce, Data, Cfg};

  std::cout << "Calculating gpu" << std::endl;
  Ty Result;
  auto Elapsed = Tester.calculate(Result);

  std::cout << "Measured time: " << Elapsed.first / msec_per_sec << std::endl
            << "Pure execution time: " << Elapsed.second / nsec_per_sec
            << std::endl;

#if defined(MEASURE_NORMAL) && defined(VERIFY)
  if (Result != ResultH) {
    std::cout << "Mismatch result: " << std::endl;
    std::cout << Result << " vs " << ResultH << std::endl;
    std::terminate();
  }
#endif
  return Tester;
}

template <typename ReductionChildT>
void test_sequence(int argc, char **argv, sycl::kernel_id kid) {
  using Ty = typename ReductionChildT::type;
  std::cout << "Welcome to reduction" << std::endl;

  try {
    auto Cfg = reduce::read_config(argc, argv);
    auto Q = set_queue();
    print_info(std::cout, Q.get_device());

    IBundleTy SrcBundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
        Q.get_context(), {kid});
    // here we can do specialization constants and many more
    OBundleTy ObjBundle = sycl::compile(SrcBundle);
    EBundleTy ExeBundle = sycl::link(ObjBundle);

    std::vector<Ty> Data;
    Data.resize(Cfg.Sz);
    constexpr Ty MAX_VAL = 10;
    if (Cfg.ValExists) {
      std::cout << "Initializing with value = " << Cfg.Val << std::endl;
      std::fill(Data.begin(), Data.end(), Cfg.Val);
    } else {
      std::cout << "Initializing with random" << std::endl;
      rand_initialize(Data.begin(), Data.end(), 0, MAX_VAL);
    }
    single_reduce_sequence<ReductionChildT>(Q, Cfg, Data.data(), ExeBundle);

  } catch (sycl::exception const &err) {
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
