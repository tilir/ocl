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

// class is used for kernel name
template <typename T> class hist_local_shared;

template <typename T>
class HistogrammLocalShared : public sycltesters::Histogramm<T> {
  using sycltesters::Histogramm<T>::Queue;
  unsigned Gsz_, Lsz_;

public:
  HistogrammLocalShared(cl::sycl::queue &DeviceQueue, unsigned Gsz,
                        unsigned Lsz)
      : sycltesters::Histogramm<T>(DeviceQueue), Gsz_(Gsz), Lsz_(Lsz) {}

  sycltesters::EvtRet_t operator()(const T *Data, T *Bins, size_t NumData,
                                   size_t NumBins) override {
    assert(Data != nullptr && Bins != nullptr);
    const auto LSZ = Lsz_; // avoid implicit capture of this
    std::vector<cl::sycl::event> ProfInfo;
    auto &DeviceQueue = Queue();
    auto *BufferData = cl::sycl::malloc_shared<T>(NumData, DeviceQueue);
    auto *BufferBins = cl::sycl::malloc_shared<T>(NumBins, DeviceQueue);
    std::copy(Data, Data + NumData, BufferData);
    std::copy(Bins, Bins + NumBins, BufferBins); // zero-out
    cl::sycl::nd_range<1> DataSz{Gsz_, Lsz_};

    if (Gsz_ < NumBins)
      throw std::runtime_error("Global size need to be more than # of bins");

    auto Evt = DeviceQueue.submit([&](cl::sycl::handler &Cgh) {
      // note local buffer is of NumBins but local iteration size is of LSZ
      using LTy = cl::sycl::accessor<T, 1, sycl_atomic, sycl_local>;
      LTy LocalHist{cl::sycl::range<1>{NumBins}, Cgh};
      auto KernHist = [BufferData, NumData, BufferBins, NumBins, LSZ,
                       LocalHist](cl::sycl::nd_item<1> WorkItem) {
        const int N = WorkItem.get_global_id(0);
        const int L = WorkItem.get_local_id(0);
        const int Group = WorkItem.get_group(0);
        const int GSZ = WorkItem.get_global_range(0);

        // zero-out
        for (int I = L; I < NumBins; I += LSZ)
          LocalHist[I].store(0);
        WorkItem.barrier(sycl_local_fence);

        // building local histograms
        for (int I = N; I < NumData; I += GSZ) {
          const T Data = BufferData[I];
          LocalHist[Data].fetch_add(1);
        }
        WorkItem.barrier(sycl_global_fence);

        // combining all local histograms
        for (int I = L; I < NumBins; I += LSZ) {
          T Data = LocalHist[I].load();
          global_atomic_ref<T>(BufferBins[I]).fetch_add(Data);
        }
      };

      Cgh.parallel_for<class hist_local_shared<T>>(DataSz, KernHist);
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
  sycltesters::test_sequence<HistogrammLocalShared<int>>(argc, argv);
}