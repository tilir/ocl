//------------------------------------------------------------------------------
//
// Histogram with simplest kernel (SYCL vs serial CPU)
// Demonstrates atomic accessor in SYCL
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
template <typename T> class hist_naive_buf;

template <typename T>
class HistogrammNaiveBuf : public sycltesters::Histogramm<T> {
  using sycltesters::Histogramm<T>::Queue;
  unsigned Gsz_, Lsz_;

public:
  HistogrammNaiveBuf(cl::sycl::queue &DeviceQueue, ConfigTy Cfg)
      : sycltesters::Histogramm<T>(DeviceQueue), Gsz_(Cfg.GlobSz),
        Lsz_(Cfg.LocSz) {}

  sycltesters::EvtRet_t operator()(const T *Data, T *Bins, int NumData,
                                   int NumBins, EBundleTy ExeBundle) override {
    assert(Data != nullptr && Bins != nullptr);
    sycltesters::EvtVec_t ProfInfo;

    // avoid memory allocation with use_host_ptr
    cl::sycl::buffer<T, 1> BufferData(Data, NumData, {host_ptr});
    cl::sycl::buffer<T, 1> BufferBins(Bins, NumBins, {host_ptr});
    cl::sycl::nd_range<1> DataSz{Gsz_, Lsz_};
    BufferData.set_final_data(nullptr);

    auto &DeviceQueue = Queue();

    auto Evt = DeviceQueue.submit([&](cl::sycl::handler &Cgh) {
      auto Data = BufferData.template get_access<sycl_read>(Cgh);
      auto Bins = BufferBins.template get_access<sycl_atomic>(Cgh);

      auto KernHist = [Data, NumData, Bins](cl::sycl::nd_item<1> WorkItem) {
        const int N = WorkItem.get_global_id(0);
        const int Gsz = WorkItem.get_global_range(0);
        for (int I = N; I < NumData; I += Gsz)
          Bins[Data[I]].fetch_add(1);
      };

      Cgh.parallel_for<class hist_naive_buf<T>>(DataSz, KernHist);
    });

    ProfInfo.emplace_back(Evt, "Calculating histogram");
    Evt.wait();

    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycl::kernel_id kid = sycl::get_kernel_id<hist_naive_buf<int>>();
  sycltesters::test_sequence<HistogrammNaiveBuf<int>>(argc, argv, kid);
}
