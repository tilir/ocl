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

// class is used for kernel name
template <typename T> class hist_naive_shared;

template <typename T>
class HistogrammNaiveShared : public sycltesters::Histogramm<T> {
  using sycltesters::Histogramm<T>::Queue;
  unsigned Gsz_, Lsz_;

public:
  HistogrammNaiveShared(cl::sycl::queue &DeviceQueue, unsigned Gsz,
                        unsigned Lsz)
      : sycltesters::Histogramm<T>(DeviceQueue), Gsz_(Gsz), Lsz_(Lsz) {}

  sycltesters::EvtRet_t operator()(const T *Data, T *Bins, size_t NumData,
                                   size_t NumBins) override {
    assert(Data != nullptr && Bins != nullptr);
    std::vector<cl::sycl::event> ProfInfo;
    auto &DeviceQueue = Queue();
    auto *BufferData = cl::sycl::malloc_shared<T>(NumData, DeviceQueue);
    auto *BufferBins = cl::sycl::malloc_shared<T>(NumBins, DeviceQueue);
    std::copy(Data, Data + NumData, BufferData);
    std::copy(Bins, Bins + NumBins, BufferBins); // zero-out
    cl::sycl::nd_range<1> DataSz{Gsz_, Lsz_};

    auto Evt = DeviceQueue.submit([&](cl::sycl::handler &Cgh) {
      auto KernHist = [BufferData, NumData,
                       BufferBins](cl::sycl::nd_item<1> WorkItem) {
        const int N = WorkItem.get_global_id(0);
        const int Gsz = WorkItem.get_global_range(0);
        for (int I = N; I < NumData; I += Gsz)
          global_atomic_ref<T>(BufferBins[BufferData[I]]).fetch_add(1);
      };

      Cgh.parallel_for<class hist_naive_shared<T>>(DataSz, KernHist);
    });

    ProfInfo.push_back(Evt);
    Evt.wait();

    // copy back
    std::copy(BufferBins, BufferBins + NumBins, Bins);

    cl::sycl::free(BufferData, DeviceQueue);
    cl::sycl::free(BufferBins, DeviceQueue);
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<HistogrammNaiveShared<int>>(argc, argv);
}