//------------------------------------------------------------------------------
//
// Histogram with simplest kernel (SYCL vs serial CPU)
// Demonstrates explicit atomic_ref usage
//
//------------------------------------------------------------------------------
//
// This file is licensed after LGPL v3
// Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
//
//------------------------------------------------------------------------------

#include <cassert>
#include <iostream>
#include <vector>

#include <CL/sycl.hpp>

#include "hist_testers.hpp"

using ConfigTy = sycltesters::hist::Config;

// class is used for kernel name
template <typename T> class hist_naive_shared;

template <typename T>
class HistogrammNaiveShared : public sycltesters::Histogramm<T> {
  using sycltesters::Histogramm<T>::Queue;
  unsigned Gsz_, Lsz_;

public:
  HistogrammNaiveShared(sycl::queue &DeviceQueue, ConfigTy Cfg)
      : sycltesters::Histogramm<T>(DeviceQueue), Gsz_(Cfg.GlobSz),
        Lsz_(Cfg.LocSz) {}

  sycltesters::EvtRet_t operator()(const T *Data, T *Bins, int NumData,
                                   int NumBins) override {
    assert(Data != nullptr && Bins != nullptr);
    const auto GSZ = Gsz_;
    sycltesters::EvtVec_t ProfInfo;
    auto &DeviceQueue = Queue();
    auto *BufferData = sycl::malloc_shared<T>(NumData, DeviceQueue);
    auto *BufferBins = sycl::malloc_shared<T>(NumBins, DeviceQueue);
    std::copy(Data, Data + NumData, BufferData);
    std::fill(BufferBins, BufferBins + NumBins, 0);
    sycl::range<1> DataSz{GSZ};

    auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
      auto KernHist = [=](sycl::id<1> Id) {
        const int N = Id.get(0);
        for (int I = N; I < NumData; I += GSZ)
          global_atomic_ref<T>(BufferBins[BufferData[I]]).fetch_add(1);
      };
      Cgh.parallel_for<class hist_naive_shared<T>>(DataSz, KernHist);
    });

    ProfInfo.emplace_back(Evt, "Calculate histogramm");
    DeviceQueue.wait();

    std::copy(BufferBins, BufferBins + NumBins, Bins);
    sycl::free(BufferData, DeviceQueue);
    sycl::free(BufferBins, DeviceQueue);
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<HistogrammNaiveShared<int>>(argc, argv);
}
