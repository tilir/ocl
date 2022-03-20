//------------------------------------------------------------------------------
//
// Generic code to test different variants of bitonic sort
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

#include <bit>
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

constexpr int DEF_SIZE = 10;

namespace sycltesters {

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
  int Step, Stage, NSeq, SeqLen, Power2;

  void SwapElements(T *Vec) {
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
    if (std::popcount(Sz) != 1 || Sz < 2)
      throw std::runtime_error("Please use only power-of-two arrays");

    int N = std::countr_zero(Sz);

    for (Step = 0; Step < N; Step++) {
      for (Stage = Step; Stage >= 0; Stage--) {
        NSeq = 1 << (N - Stage - 1);
        SeqLen = 1 << (Stage + 1);
        Power2 = 1 << (Step - Stage);
        SwapElements(Vec);
      }
    }
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
    for (int i = 0; i < Sz_; i++) {
      A_[i] = d();
#ifdef VISUALIZE
      std::cout << A_[i] << " ";
#endif
    }
#ifdef VISUALIZE
    std::cout << std::endl;
#endif
  }

  std::pair<unsigned, unsigned> calculate() {
    unsigned EvtTiming = 0;
    Timer_.start();
    EvtRet_t Ret = Sorter_(A_.data(), Sz_);
    EvtTiming += getTime(Ret);
    Timer_.stop();
#ifdef VISUALIZE
    for (int i = 0; i < Sz_; i++)
      std::cout << A_[i] << " ";
    std::cout << std::endl;
#endif
#ifdef VERIFY
    if (!std::is_sorted(A_.begin(), A_.end()))
      throw std::runtime_error("Sorting failed");
#endif
    return {Timer_.elapsed(), EvtTiming};
  }
};

template <typename BitonicChildT> void test_sequence(int argc, char **argv) {
  std::cout << "Welcome to vector addition" << std::endl;

  try {
    unsigned Size = 0, NReps = 0;

    optparser_t OptParser;
    OptParser.parse(argc, argv);

    Size = OptParser.template get<int>("size");

    if (Size > 31)
      throw std::runtime_error("Size is logarithmic, 31 is max");

    if (Size == 0)
      Size = DEF_SIZE;

    std::cout << "Using vector size = " << (1 << Size) << std::endl;

    auto Q = set_queue();
    print_info(std::cout, Q.get_device());

#ifdef MEASURE_NORMAL
    BitonicSortHost<int> VaddH{Q}; // Q unused for this derived class
    BitonicSortTester<int> TesterH{VaddH, (1 << Size)};
    TesterH.initialize();
    auto ElapsedH = TesterH.calculate();
    std::cout << "Measured host time: " << ElapsedH.first << std::endl;
#endif

    BitonicChildT BitonicSort{Q};
    using Ty = typename BitonicChildT::type;
    BitonicSortTester<Ty> Tester{BitonicSort, 1 << Size};

    std::cout << "Initializing" << std::endl;
    Tester.initialize();

    std::cout << "Calculating" << std::endl;
    auto Elapsed = Tester.calculate();

    std::cout << "Measured time: " << Elapsed.first / 1000.0 << std::endl;
    std::cout << "Pure execution time: " << Elapsed.second / 1000000000.0
              << std::endl;
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
