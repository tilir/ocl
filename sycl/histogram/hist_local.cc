//------------------------------------------------------------------------------
//
// Histogram with local memory (SYCL vs serial CPU).
// In this example SYCL uses both global and local memory.
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
template <typename T> class hist_local_shared;

template <typename T>
class HistogrammLocalShared : public sycltesters::Histogramm<T> {
  using sycltesters::Histogramm<T>::Queue;
  unsigned Gsz_, Lsz_;

public:
  HistogrammLocalShared(sycl::queue &DeviceQueue, ConfigTy Cfg)
      : sycltesters::Histogramm<T>(DeviceQueue), Gsz_(Cfg.GlobSz),
        Lsz_(Cfg.LocSz) {}

  sycltesters::EvtRet_t operator()(const T *Data, T *Bins, int NumData,
                                   int NumBins) override {
    assert(Data != nullptr && Bins != nullptr);
    const auto LSZ = Lsz_;
    const auto GSZ = Gsz_;
    sycltesters::EvtVec_t ProfInfo;
    auto &DeviceQueue = Queue();
    auto *BufferData = sycl::malloc_shared<T>(NumData, DeviceQueue);
    auto *BufferBins = sycl::malloc_shared<T>(NumBins, DeviceQueue);
    std::copy(Data, Data + NumData, BufferData);
    std::fill(BufferBins, BufferBins + NumBins, 0);

    // note local buffer is of NumBins but local iteration size is of LSZ
    using LTy = sycl::accessor<T, 1, sycl_read_write, sycl_local>;
    sycl::range<1> LocalMemorySize{NumBins};
    sycl::nd_range<1> DataSz{GSZ, LSZ};

    auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
      LTy LocalHist{LocalMemorySize, Cgh};
      auto KernHist = [=](sycl::nd_item<1> WorkItem) {
        const int N = WorkItem.get_global_id(0);
        const int L = WorkItem.get_local_id(0);

        // zero-out local memory
        for (int I = L; I < NumBins; I += LSZ)
          LocalHist[I] = 0;
        WorkItem.barrier(sycl_local_fence);

        // building local histograms
        for (int I = N; I < NumData; I += GSZ) {
          const T Data = BufferData[I];
          local_atomic_ref<T>(LocalHist[Data]).fetch_add(1);
        }
        WorkItem.barrier(sycl_local_fence);

        // combining all local histograms
        for (int I = L; I < NumBins; I += LSZ) {
          const T Data = LocalHist[I];
          global_atomic_ref<T>(BufferBins[I]).fetch_add(Data);
        }
      };

      Cgh.parallel_for<class hist_local_shared<T>>(DataSz, KernHist);
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
  sycltesters::test_sequence<HistogrammLocalShared<int>>(argc, argv);
}
