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
  HistogrammNaiveBuf(sycl::queue &DeviceQueue, ConfigTy Cfg)
      : sycltesters::Histogramm<T>(DeviceQueue), Gsz_(Cfg.GlobSz),
        Lsz_(Cfg.LocSz) {}

  sycltesters::EvtRet_t operator()(const T *Data, T *Bins, int NumData,
                                   int NumBins) override {
    assert(Data != nullptr && Bins != nullptr);
    sycltesters::EvtVec_t ProfInfo;
    const auto GSZ = Gsz_;

#ifdef HOST_PTR
    // avoid memory allocations but at the cost of host access every time
    sycl::buffer<T, 1> BufferData(Data, NumData, {host_ptr});
    sycl::buffer<T, 1> BufferBins(Bins, NumBins, {host_ptr});
#else
    sycl::buffer<T, 1> BufferData(Data, NumData);
    sycl::buffer<T, 1> BufferBins(Bins, NumBins);
#endif
    sycl::range<1> DataSz{GSZ};
    BufferData.set_final_data(nullptr);

    auto &DeviceQueue = Queue();

    auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
      auto Data = BufferData.template get_access<sycl_read>(Cgh);
      auto Bins = BufferBins.template get_access<sycl_atomic>(Cgh);

      auto KernHist = [=](sycl::id<1> Id) {
        const int N = Id.get(0);
        for (int I = N; I < NumData; I += GSZ)
          Bins[Data[I]].fetch_add(1);
      };

      Cgh.parallel_for<class hist_naive_buf<T>>(DataSz, KernHist);
    });

    ProfInfo.emplace_back(Evt, "Calculating histogram");
    DeviceQueue.wait(); // or explicit host accessor to NumBins
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<HistogrammNaiveBuf<int>>(argc, argv);
}
