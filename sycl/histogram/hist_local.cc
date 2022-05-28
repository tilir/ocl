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
  HistogrammLocalShared(cl::sycl::queue &DeviceQueue, ConfigTy Cfg)
      : sycltesters::Histogramm<T>(DeviceQueue), Gsz_(Cfg.GlobSz),
        Lsz_(Cfg.LocSz) {}

  sycltesters::EvtRet_t operator()(const T *Data, T *Bins, int NumData,
                                   int NumBins, EBundleTy ExeBundle) override {
    assert(Data != nullptr && Bins != nullptr);
    const auto LSZ = Lsz_; // avoid implicit capture of this
    const size_t LMEM = NumBins;
    sycltesters::EvtVec_t ProfInfo;
    auto &DeviceQueue = Queue();
    auto *BufferData = sycl::malloc_shared<T>(NumData, DeviceQueue);
    auto *BufferBins = sycl::malloc_shared<T>(NumBins, DeviceQueue);

    auto EvtCpyData = DeviceQueue.copy(Data, BufferData, NumData);
    auto EvtFillBins = DeviceQueue.copy(Bins, BufferBins, NumBins);
    sycl::nd_range<1> DataSz{Gsz_, Lsz_};
    ProfInfo.emplace_back(EvtCpyData, "Copy Data");
    ProfInfo.emplace_back(EvtFillBins, "Zero-out Bins");

    auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
      Cgh.depends_on(EvtCpyData);
      Cgh.depends_on(EvtFillBins);
      Cgh.use_kernel_bundle(ExeBundle);

      // note local buffer is of NumBins but local iteration size is of LSZ
      using LTy = sycl::accessor<T, 1, sycl_read_write, sycl_local>;
      sycl::range<1> LocalMemorySize{LMEM};
      LTy LocalHist{LocalMemorySize, Cgh};
      auto KernHist = [=](sycl::nd_item<1> WorkItem) {
        const int N = WorkItem.get_global_id(0);
        const int L = WorkItem.get_local_id(0);
        const int Group = WorkItem.get_group(0);
        const int GSZ = WorkItem.get_global_range(0);

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

    // copy back (note dependency on Evt)
    auto EvtCpyBins = DeviceQueue.copy(BufferBins, Bins, NumBins, Evt);
    ProfInfo.emplace_back(EvtCpyBins, "Copy bins back");
    DeviceQueue.wait();

    cl::sycl::free(BufferData, DeviceQueue);
    cl::sycl::free(BufferBins, DeviceQueue);
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycl::kernel_id kid = sycl::get_kernel_id<hist_local_shared<int>>();
  sycltesters::test_sequence<HistogrammLocalShared<int>>(argc, argv, kid);
}
