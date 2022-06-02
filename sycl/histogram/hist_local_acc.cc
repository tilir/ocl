//------------------------------------------------------------------------------
//
// Histogram with local memory (SYCL vs serial CPU).
// In this example SYCL uses both global and local accessors.
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
template <typename T> class hist_local_acc;

template <typename T>
class HistogrammLocalAcc : public sycltesters::Histogramm<T> {
  using sycltesters::Histogramm<T>::Queue;
  unsigned Gsz_, Lsz_;

public:
  HistogrammLocalAcc(sycl::queue &DeviceQueue, ConfigTy Cfg)
      : sycltesters::Histogramm<T>(DeviceQueue), Gsz_(Cfg.GlobSz),
        Lsz_(Cfg.LocSz) {}

  sycltesters::EvtRet_t operator()(const T *Data, T *Bins, int NumData,
                                   int NumBins) override {
    assert(Data != nullptr && Bins != nullptr);
    const auto GSZ = Gsz_;
    const auto LSZ = Lsz_;
    const size_t LMEM = NumBins;
    sycltesters::EvtVec_t ProfInfo;
    auto &DeviceQueue = Queue();

    sycl::buffer<T, 1> BufferData(Data, NumData);
    sycl::buffer<T, 1> BufferBins(Bins, NumBins);

    // note local buffer is of NumBins but local iteration size is of LSZ
    using LTy = sycl::accessor<T, 1, sycl_atomic, sycl_local>;
    sycl::range<1> LocalMemorySize{LMEM};
    sycl::nd_range<1> DataSz{GSZ, LSZ};

    auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
      auto Data = BufferData.template get_access<sycl_read>(Cgh);
      auto Bins = BufferBins.template get_access<sycl_atomic>(Cgh);

      LTy LocalHist{LocalMemorySize, Cgh};
      auto KernHist = [=](sycl::nd_item<1> WorkItem) {
        const int N = WorkItem.get_global_id(0);
        const int L = WorkItem.get_local_id(0);

        // zero-out local memory
        for (int I = L; I < NumBins; I += LSZ)
          LocalHist[I].store(0);
        WorkItem.barrier(sycl_local_fence);

        // building local histograms
        for (int I = N; I < NumData; I += GSZ)
          LocalHist[Data[I]].fetch_add(1);
        WorkItem.barrier(sycl_local_fence);

        // combining all local histograms
        for (int I = L; I < NumBins; I += LSZ) {
          const T LocalData = LocalHist[I].load();
          Bins[I].fetch_add(LocalData);
        }
      };

      Cgh.parallel_for<class hist_local_acc<T>>(DataSz, KernHist);
    });

    ProfInfo.emplace_back(Evt, "Calculate histogramm");
    DeviceQueue.wait();
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<HistogrammLocalAcc<int>>(argc, argv);
}
