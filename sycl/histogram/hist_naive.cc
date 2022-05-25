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
  HistogrammNaiveShared(cl::sycl::queue &DeviceQueue, ConfigTy Cfg)
      : sycltesters::Histogramm<T>(DeviceQueue), Gsz_(Cfg.GlobSz),
        Lsz_(Cfg.LocSz) {}

  sycltesters::EvtRet_t operator()(const T *Data, T *Bins, int NumData,
                                   int NumBins, EBundleTy ExeBundle) override {
    assert(Data != nullptr && Bins != nullptr);
    sycltesters::EvtVec_t ProfInfo;
    auto &DeviceQueue = Queue();
    auto *BufferData = cl::sycl::malloc_shared<T>(NumData, DeviceQueue);
    auto *BufferBins = cl::sycl::malloc_shared<T>(NumBins, DeviceQueue);

    auto EvtCpyData = DeviceQueue.copy(Data, BufferData, NumData);
    auto EvtFillBins = DeviceQueue.copy(Bins, BufferBins, NumBins);
    cl::sycl::nd_range<1> DataSz{Gsz_, Lsz_};
    ProfInfo.emplace_back(EvtCpyData, "Copy Data");
    ProfInfo.emplace_back(EvtFillBins, "Zero-out Bins");

    auto Evt = DeviceQueue.submit([&](cl::sycl::handler &Cgh) {
      Cgh.depends_on(EvtCpyData);
      Cgh.depends_on(EvtFillBins);
      Cgh.use_kernel_bundle(ExeBundle);

      auto KernHist = [BufferData, NumData,
                       BufferBins](cl::sycl::nd_item<1> WorkItem) {
        const int N = WorkItem.get_global_id(0);
        const int Gsz = WorkItem.get_global_range(0);
        for (int I = N; I < NumData; I += Gsz)
          global_atomic_ref<T>(BufferBins[BufferData[I]]).fetch_add(1);
      };

      Cgh.parallel_for<class hist_naive_shared<T>>(DataSz, KernHist);
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
  sycl::kernel_id kid = sycl::get_kernel_id<hist_naive_shared<int>>();
  sycltesters::test_sequence<HistogrammNaiveShared<int>>(argc, argv, kid);
}
