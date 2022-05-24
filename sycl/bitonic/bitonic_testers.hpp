//------------------------------------------------------------------------------
//
// Generic code to test different variants of bitonic sort
// Avoiding tons of boilerplate otherwise
//
// Macros to control things:
//  * inherited from testers.hpp: RUNHOST, INORD...
//  -DVERIFY            : check for sorted
//  -DMEASURE_NORMAL    : check against CPU sort (default is std::sort)
//  -DCHECK_BITONIC_CPU : check against bitonic sort CPU code
//
//------------------------------------------------------------------------------
//
// This file is licensed after LGPL v3
// Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
//
//------------------------------------------------------------------------------

#pragma once

#include <algorithm>
#include <bit>
#include <cassert>
#include <chrono>
#include <iostream>
#include <iterator>
#include <vector>

#include <CL/sycl.hpp>

// problems with boost in OneAPI console on Windows
#ifdef USE_BOOST_OPTPARSE
#include "optparse.hpp"
#else
#include "optparse_alt.hpp"
#endif

#include "testers.hpp"

constexpr int DEF_SIZE = 10;
constexpr int DEF_BLOCK_SIZE = 1;

namespace sycltesters {

namespace bitonicsort {
struct Config {
  unsigned Size, LocalSize;
  int Vis = 0, Quiet;
};

inline Config read_config(int argc, char **argv) {
  Config Cfg;
  options::Parser OptParser;
  OptParser.template add<int>(
      "size", DEF_SIZE, "logarithmic size to sort (1 << size) is real size");
  OptParser.template add<int>("lsz", DEF_BLOCK_SIZE, "local size");
  OptParser.template add<int>("vis", 0,
                              "pass 1 to visualize before and after sort");
  OptParser.template add<int>("q", 0, "pass 1 for quiet mode");
  OptParser.parse(argc, argv);

  Cfg.Size = OptParser.template get<int>("size");
  Cfg.LocalSize = OptParser.template get<int>("lsz");
  Cfg.Quiet = OptParser.template get<int>("q");
  Cfg.Vis = OptParser.template get<int>("vis");

  if (Cfg.Size < 2 || Cfg.Size > 31) {
    std::cerr << "Size is logarithmic, 2 is min, 31 is max" << std::endl;
    std::terminate();
  }

  if (Cfg.Quiet && Cfg.Vis) {
    std::cerr << "Please select quiet or visual" << std::endl;
    std::terminate();
  }

  return Cfg;
}
} // namespace bitonicsort

template <typename T> class BitonicSort {
  cl::sycl::queue DeviceQueue_;

public:
  using type = T;
  BitonicSort(cl::sycl::queue &DeviceQueue) : DeviceQueue_(DeviceQueue) {}
  virtual EvtRet_t operator()(T *Vec, size_t Sz) = 0;
  cl::sycl::queue &Queue() { return DeviceQueue_; }
  const cl::sycl::queue &Queue() const { return DeviceQueue_; }
  virtual ~BitonicSort() {}
};

template <typename T> struct BitonicSortHost : public BitonicSort<T> {
  void SwapElements(T *Vec, int NSeq, int SeqLen, int Power2) {
    for (int SNum = 0; SNum < NSeq; SNum++) {
      int Odd = SNum / Power2;
      bool Increasing = ((Odd % 2) == 0);
      int HalfLen = SeqLen / 2;

      // For all elements in a bitonic sequence, swap them if needed
      for (int I = SNum * SeqLen; I < SNum * SeqLen + HalfLen; I++) {
        int J = I + HalfLen;
        if (((Vec[I] > Vec[J]) && Increasing) ||
            ((Vec[I] < Vec[J]) && !Increasing))
          std::swap(Vec[I], Vec[J]);
      }
    }
  }

public:
  BitonicSortHost(cl::sycl::queue &DeviceQueue) : BitonicSort<T>(DeviceQueue) {}
  EvtRet_t operator()(T *Vec, size_t Sz) override {
    assert(Vec);
    int NSeq, SeqLen, Step, Stage, Power2;
#if CHECK_BITONIC_CPU
    if (std::popcount(Sz) != 1 || Sz < 2)
      throw std::runtime_error("Please use only power-of-two arrays");

    int N = std::countr_zero(Sz);

    for (Step = 0; Step < N; Step++) {
      for (Stage = Step; Stage >= 0; Stage--) {
        NSeq = 1 << (N - Stage - 1);
        SeqLen = 1 << (Stage + 1);
        Power2 = 1 << (Step - Stage);
        SwapElements(Vec, NSeq, SeqLen, Power2);
      }
    }
#else
    std::sort(Vec, Vec + Sz);
#endif
    return {}; // nothing to construct as event
  }
};

template <typename T> class BitonicSortTester {
  BitonicSort<T> &Sorter_;
  Timer Timer_;
  unsigned Sz_;
  std::vector<T> A_;

public:
  BitonicSortTester(BitonicSort<T> &Sorter, unsigned Sz)
      : Sorter_(Sorter), Sz_(Sz), A_(Sz) {}

  void initialize() {
    Dice d(0, Sz_);
    std::generate(A_.begin(), A_.end(), [&] { return d(); });
  }

  std::pair<unsigned, unsigned> calculate() {
    unsigned EvtTiming = 0;
    Timer_.start();
    EvtRet_t Ret = Sorter_(A_.data(), Sz_);
    EvtTiming += getTime(Ret);
    Timer_.stop();
    return {Timer_.elapsed(), EvtTiming};
  }

  auto begin() { return A_.begin(); }
  auto end() { return A_.end(); }
};

template <typename BitonicChildT> void test_sequence(int argc, char **argv) {
  bool Quiet;
  try {
    auto Cfg = bitonicsort::read_config(argc, argv);
    Quiet = Cfg.Quiet;
    auto Q = set_queue();
    if (!Quiet) {
      std::cout << "Welcome to bitonic sort" << std::endl;
      std::cout << "Using vector size = " << (1 << Cfg.Size) << std::endl;
      print_info(std::cout, Q.get_device());
    }

    using Ty = typename BitonicChildT::type;
#ifdef MEASURE_NORMAL
    BitonicSortHost<Ty> BitonicSortH{Q}; // Q unused for this derived class
    BitonicSortTester<Ty> TesterH{BitonicSortH, 1u << Cfg.Size};
    TesterH.initialize();
    auto ElapsedH = TesterH.calculate();
    if (!Quiet)
      std::cout << "Measured host time: " << ElapsedH.first << std::endl;
#endif

    BitonicChildT BitonicSort{Q, Cfg.LocalSize};
    BitonicSortTester<Ty> Tester{BitonicSort, 1u << Cfg.Size};

    if (!Quiet)
      std::cout << "Initializing" << std::endl;
    Tester.initialize();

    if (Cfg.Vis) {
      std::cout << "Before sort:" << std::endl;
      visualize_seq(Tester.begin(), Tester.end(), std::cout);
    }

    if (!Quiet)
      std::cout << "Calculating" << std::endl;
    auto Elapsed = Tester.calculate();

    if (Cfg.Vis) {
      std::cout << "After sort:" << std::endl;
      visualize_seq(Tester.begin(), Tester.end(), std::cout);
    }

#ifdef VERIFY
    if (!std::is_sorted(Tester.begin(), Tester.end())) {
      std::cerr << "Sorting failed" << std::endl;
      std::terminate();
    }
#endif

    if (!Quiet) {
      std::cout << "Measured time: " << Elapsed.first / msec_per_sec
                << std::endl;
      std::cout << "Pure execution time: " << Elapsed.second / nsec_per_sec
                << std::endl;
    } else {
      std::cout << Cfg.Size << " " << Elapsed.first / msec_per_sec << std::endl;
    }
  } catch (cl::sycl::exception const &err) {
    std::cerr << "SYCL ERROR: " << err.what() << "\n";
    std::terminate();
  } catch (std::exception const &err) {
    std::cerr << "Exception: " << err.what() << "\n";
    std::terminate();
  } catch (...) {
    std::cerr << "Unknown error\n";
    std::terminate();
  }
  if (!Quiet)
    std::cout << "Everything is correct" << std::endl;
}

} // namespace sycltesters
