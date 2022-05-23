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

// class is used for kernel name
template <typename T> class hist_naive_buf;

template <typename T>
class HistogrammNaiveBuf : public sycltesters::Histogramm<T> {
  using sycltesters::Histogramm<T>::Queue;
  unsigned Gsz_, Lsz_;

public:
  HistogrammNaiveBuf(cl::sycl::queue &DeviceQueue, unsigned Gsz, unsigned Lsz)
      : sycltesters::Histogramm<T>(DeviceQueue), Gsz_(Gsz), Lsz_(Lsz) {}

  sycltesters::EvtRet_t operator()(const T *Data, T *Bins, size_t NumData,
                                   size_t NumBins,
                                   EBundleTy ExeBundle) override {
    assert(Data != nullptr && Bins != nullptr);
    sycltesters::EvtVec_t ProfInfo;
    cl::sycl::buffer<T, 1> BufferData(Data, NumData);
    cl::sycl::buffer<T, 1> BufferBins(Bins, NumBins);
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

    ProfInfo.push_back(Evt);
    Evt.wait();

    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycl::kernel_id kid = sycl::get_kernel_id<hist_naive_buf<int>>();
  sycltesters::test_sequence<HistogrammNaiveBuf<int>>(argc, argv, kid);
}