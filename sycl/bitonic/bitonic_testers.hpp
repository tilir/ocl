//------------------------------------------------------------------------------
//
// Generic code to test different variants of bitonic sort
// Avoiding tons of boilerplate otherwise
//
// Macros to control things:
//  * inherited from testers.hpp: RUNHOST, INORD...
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
#include <numeric>
#include <vector>

#include <CL/sycl.hpp>

// problems with boost in OneAPI console on Windows
#ifdef USE_BOOST_OPTPARSE
#include "optparse.hpp"
#else
#include "optparse_alt.hpp"
#endif

#include "testers.hpp"

constexpr int DEF_SIZE = 20;
constexpr int DEF_BLOCK_SIZE = 256;

namespace sycltesters {

namespace bitonicsort {
struct Config {
  std::string FileName;
  unsigned Size, LocSz;
  bool Vis = false, Quiet = false, Detailed = false, Definit = false,
       Verbose = false, InpFile = false;
};

inline Config read_config(int argc, char **argv) {
  Config Cfg;
  options::Parser OptParser;
  OptParser.template add<int>(
      "size", DEF_SIZE, "logarithmic size to sort (1 << size) is real size");
  OptParser.template add<int>("lsz", DEF_BLOCK_SIZE, "local size");
  OptParser.template add<int>("vis", 0, "visualize before and after sort");
  OptParser.template add<int>("detailed", 0, "detailed events");
  OptParser.template add<int>("quiet", 0, "quiet mode for bulk runs");
  OptParser.template add<int>("definit", 0, "initialize for worst case");
  OptParser.template add<std::string>("inpfile", "",
                                      "initialize from given file");
  OptParser.template add<int>("verbose", 0,
                              "really verbose mode: after each step");
  OptParser.parse(argc, argv);

  Cfg.Size = OptParser.template get<int>("size");
  Cfg.LocSz = OptParser.template get<int>("lsz");
  Cfg.Vis = OptParser.exists("vis");
  Cfg.Detailed = OptParser.exists("detailed");
  Cfg.Definit = OptParser.exists("definit");
  Cfg.Verbose = OptParser.exists("verbose");
  Cfg.InpFile = OptParser.exists("inpfile");
  Cfg.FileName = OptParser.template get<std::string>("inpfile");

  if (Cfg.Size < 2 || Cfg.Size > 31)
    throw std::runtime_error("Size is logarithmic, 2 is min, 31 is max");

  if (OptParser.exists("quiet")) {
    Cfg.Quiet = true;
    Cfg.Vis = false; // quiet implies novis
    qout.set(Cfg.Quiet);
  }

  return Cfg;
}
} // namespace bitonicsort

template <typename T> class BitonicSort {
  sycl::queue DeviceQueue_;

public:
  using type = T;
  BitonicSort(sycl::queue &DeviceQueue) : DeviceQueue_(DeviceQueue) {}
  virtual EvtRet_t operator()(T *Vec, size_t Sz) = 0;
  sycl::queue &Queue() { return DeviceQueue_; }
  const sycl::queue &Queue() const { return DeviceQueue_; }
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
  BitonicSortHost(sycl::queue &DeviceQueue) : BitonicSort<T>(DeviceQueue) {}
  EvtRet_t operator()(T *Vec, size_t Sz) override {
    assert(Vec);
#if CHECK_BITONIC_CPU
    int Step, Stage;
    if (std::popcount(Sz) != 1 || Sz < 2)
      throw std::runtime_error("Please use only power-of-two arrays");

    int N = std::countr_zero(Sz);

    for (Step = 0; Step < N; Step++) {
      for (Stage = Step; Stage >= 0; Stage--) {
        int NSeq, SeqLen, Power2;
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
  bitonicsort::Config Cfg_;
  unsigned Sz_;
  std::vector<T> A_;

public:
  BitonicSortTester(BitonicSort<T> &Sorter, bitonicsort::Config Cfg)
      : Sorter_(Sorter), Cfg_(Cfg), Sz_(1u << Cfg_.Size), A_(Sz_) {}

  void initialize() {
    // deterministic worst case
    if (Cfg_.Definit) {
      std::iota(A_.rbegin(), A_.rend(), 0);
      return;
    }

    // input from file
    if (Cfg_.InpFile) {
      qout << "Reading: " << Cfg_.FileName << std::endl;
      std::ifstream If(Cfg_.FileName);
      std::istream_iterator<T> In{If}, InS;
      std::copy(In, InS, A_.begin());
      return;
    }

    // random generate
    Dice d(0, Sz_);
    std::generate(A_.begin(), A_.end(), [&] { return d(); });
  }

  template <typename It> void assign(It begin, It end) {
    A_.assign(begin, end);
  }

  std::pair<unsigned, unsigned long long> calculate() {
    Timer_.start();
    EvtRet_t Ret = Sorter_(A_.data(), A_.size());
    auto EvtTiming = getTime(Ret, Cfg_.Detailed ? false : true);
    Timer_.stop();
    return {Timer_.elapsed(), EvtTiming};
  }

  auto begin() { return A_.begin(); }
  auto end() { return A_.end(); }
};

template <typename BitonicChildT> void test_sequence(int argc, char **argv) {
  try {
    auto Cfg = bitonicsort::read_config(argc, argv);
    auto Q = set_queue();
    qout << "Welcome to bitonic sort\n";
    qout << "Using vector size = " << (1 << Cfg.Size) << "\n";
    print_info(qout, Q.get_device());

    using Ty = typename BitonicChildT::type;
    BitonicChildT BitonicSort{Q, Cfg};
    BitonicSortTester<Ty> Tester{BitonicSort, Cfg};

    qout << "Initializing\n";
    Tester.initialize();

    if (Cfg.Vis) {
      qout << "Before sort:\n";
      visualize_seq(Tester.begin(), Tester.end(), qout);
    }

#ifdef MEASURE_NORMAL
    BitonicSortHost<Ty> BitonicSortH{Q}; // Q unused for this derived class
    BitonicSortTester<Ty> TesterH{BitonicSortH, Cfg};
    TesterH.assign(Tester.begin(), Tester.end());
    auto ElapsedH = TesterH.calculate();
    qout << "Measured host time: " << ElapsedH.first << "\n";
    if (Cfg.Vis) {
      qout << "After sort (host):\n";
      visualize_seq(TesterH.begin(), TesterH.end(), qout);
    }
#endif

    qout << "Calculating\n";
    auto Elapsed = Tester.calculate();

    if (Cfg.Vis) {
      qout << "After sort:\n";
      visualize_seq(Tester.begin(), Tester.end(), qout);
    }

#ifdef VERIFY
    if (!std::is_sorted(Tester.begin(), Tester.end())) {
      std::cerr << "Sorting failed\n";
      std::terminate();
    }
// we may also check with host results
#ifdef MEASURE_NORMAL
    auto MisPoint =
        std::mismatch(TesterH.begin(), TesterH.end(), Tester.begin());
    if (MisPoint.first != TesterH.end()) {
      ptrdiff_t I = std::distance(MisPoint.first, TesterH.begin());
      std::cerr << "Mismatch at: " << I << std::endl;
      std::cerr << *MisPoint.first << " vs " << *MisPoint.second << std::endl;
      throw std::runtime_error("Mismath");
    }
#endif
#endif
    qout << "Measured time: " << Elapsed.first / msec_per_sec << "\n";

    auto ExecTime = Elapsed.second / nsec_per_sec;
    qout << "Pure execution time: " << ExecTime << "\n";

    // Quiet mode output: size, elapsed time
    if (Cfg.Quiet) {
      qout.set(!Cfg.Quiet);
      qout << Cfg.Size << " " << ExecTime << "\n";
      qout.set(Cfg.Quiet);
    }
  } catch (sycl::exception const &err) {
    std::cerr << "SYCL ERROR: " << err.what() << "\n";
    std::terminate();
  } catch (std::exception const &err) {
    std::cerr << "Exception: " << err.what() << "\n";
    std::terminate();
  } catch (...) {
    std::cerr << "Unknown error\n";
    std::terminate();
  }
  qout << "Everything is correct\n";
}

} // namespace sycltesters
