//------------------------------------------------------------------------------
//
// Histogram with big private memory blocks (SYCL vs serial CPU).
// In this example SYCL uses too much private memory
//
// > histogram\hist_private.exe -sz=200000 -hsz=1024
//
// really slow due to spills
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
template <typename T> class hist_private_shared;

template <typename T>
class HistogrammPrivateShared : public sycltesters::Histogramm<T> {
  using sycltesters::Histogramm<T>::Queue;
  unsigned Gsz_, Lsz_;

public:
  HistogrammPrivateShared(sycl::queue &DeviceQueue, ConfigTy Cfg)
      : sycltesters::Histogramm<T>(DeviceQueue), Gsz_(Cfg.GlobSz),
        Lsz_(Cfg.LocSz) {}

  sycltesters::EvtRet_t operator()(const T *Data, T *Bins, int NumData,
                                   int NumBins) override {
    assert(Data != nullptr && Bins != nullptr);
    constexpr int MAX_HSZ = 4096;
    if ((MAX_HSZ < NumBins) || ((Gsz_ % NumBins) != 0)) {
      std::cerr << "Now this example works only if #Bins <= " << MAX_HSZ
                << std::endl;
      std::cerr << "Also #Bins shall divide global size: #Bins = " << NumBins
                << ", GSZ = " << Gsz_ << std::endl;
      std::terminate();
    }
    const auto NGSZ = Gsz_ / NumBins;
    sycltesters::EvtVec_t ProfInfo;
    auto &DeviceQueue = Queue();
    auto *BufferData = sycl::malloc_shared<T>(NumData, DeviceQueue);
    auto *BufferBins = sycl::malloc_shared<T>(NumBins, DeviceQueue);
    std::copy(Data, Data + NumData, BufferData);
    std::fill(BufferBins, BufferBins + NumBins, 0);
    sycl::range<1> DataSz{NGSZ};

    auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
      auto KernHist = [=](sycl::id<1> Id) {
        T PrivateHist[MAX_HSZ] = {0};
        const int N = Id.get(0);

        // building private histograms
        for (int I = N; I < NumData / NumBins; I += NGSZ)
          for (int J = 0; J < NumBins; J += 1) {
            const T Data = BufferData[I * NumBins + J];
            PrivateHist[Data] += 1;
          }

        // combining all private histograms
        for (int I = 0; I < NumBins; I += 1) {
          const T Data = PrivateHist[I];
          global_atomic_ref<T>(BufferBins[I]).fetch_add(Data);
        }
      };

      Cgh.parallel_for<class hist_private_shared<T>>(DataSz, KernHist);
    });

    ProfInfo.emplace_back(Evt, "Calculate histogramm");

    // copy back (note dependency on Evt)
    auto EvtCpyBins = DeviceQueue.copy(BufferBins, Bins, NumBins, Evt);
    ProfInfo.emplace_back(EvtCpyBins, "Copy bins back");
    DeviceQueue.wait();

    sycl::free(BufferData, DeviceQueue);
    sycl::free(BufferBins, DeviceQueue);
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<HistogrammPrivateShared<int>>(argc, argv);
}
