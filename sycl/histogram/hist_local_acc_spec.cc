//------------------------------------------------------------------------------
//
// Histogram with local memory (SYCL vs serial CPU).
// This example demonstrates specialization constants.
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
template <typename T> class hist_local_shared_spec;

const static sycl::specialization_id<int> LSZC;
const static sycl::specialization_id<int> GSZC;

template <typename T>
class HistogrammLocalAccSpec : public sycltesters::Histogramm<T> {
  using sycltesters::Histogramm<T>::Queue;
  unsigned Gsz_, Lsz_;

public:
  HistogrammLocalAccSpec(sycl::queue &DeviceQueue, ConfigTy Cfg)
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

    sycl::kernel_id KId = sycl::get_kernel_id<hist_local_shared_spec<T>>();
    sycl::kernel_bundle KbSrc =
        sycl::get_kernel_bundle<sycl::bundle_state::input>(
            DeviceQueue.get_context(), {KId});
    KbSrc.template set_specialization_constant<LSZC>(LSZ);
    KbSrc.template set_specialization_constant<GSZC>(GSZ);
    sycl::kernel_bundle Kb = sycl::build(KbSrc);

    auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
      Cgh.use_kernel_bundle(Kb);
      auto Data = BufferData.template get_access<sycl_read>(Cgh);
      auto Bins = BufferBins.template get_access<sycl_atomic>(Cgh);

      LTy LocalHist{LocalMemorySize, Cgh};
      auto KernHist = [=](sycl::nd_item<1> WorkItem, sycl::kernel_handler Kh) {
        const int N = WorkItem.get_global_id(0);
        const int L = WorkItem.get_local_id(0);
        const int LSZK = Kh.template get_specialization_constant<LSZC>();
        const int GSZK = Kh.template get_specialization_constant<GSZC>();

        // zero-out local memory
        for (int I = L; I < NumBins; I += LSZK)
          LocalHist[I].store(0);
        WorkItem.barrier(sycl_local_fence);

        // building local histograms
        for (int I = N; I < NumData; I += GSZK)
          LocalHist[Data[I]].fetch_add(1);
        WorkItem.barrier(sycl_local_fence);

        // combining all local histograms
        for (int I = L; I < NumBins; I += LSZK) {
          const T LocalData = LocalHist[I].load();
          Bins[I].fetch_add(LocalData);
        }
      };

      Cgh.parallel_for<class hist_local_shared_spec<T>>(DataSz, KernHist);
    });

    ProfInfo.emplace_back(Evt, "Calculate histogramm");
    DeviceQueue.wait();
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<HistogrammLocalAccSpec<int>>(argc, argv);
}
