//------------------------------------------------------------------------------
//
// Reduction, naive approach (SYCL vs serial CPU).
// In this example SYCL does pretty naive reduction (not too naive yet).
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

#include "reduction_testers.hpp"

using ConfigTy = sycltesters::reduce::Config;

// class is used for kernel name
template <typename T> class reduce_naive_buf;

template <typename T>
class ReductionNaiveBuf : public sycltesters::Reduction<T> {
  using sycltesters::Reduction<T>::Queue;
  using sycltesters::Reduction<T>::Bundle;
  ConfigTy Cfg_;

public:
  ReductionNaiveBuf(sycl::queue &DeviceQueue, EBundleTy ExeBundle, ConfigTy Cfg)
      : sycltesters::Reduction<T>(DeviceQueue, ExeBundle), Cfg_(Cfg) {}

  sycltesters::EvtRet_t operator()(const T *Data, size_t NumData,
                                   T &Result) override {
    const auto GSZ = Cfg_.GlobSz;
    const auto LSZ = Cfg_.LocSz;
    const auto NumRes = GSZ / LSZ;
    assert(Data != nullptr && Bins != nullptr);
    sycltesters::EvtVec_t ProfInfo;

    // avoid memory allocation with use_host_ptr
    sycl::buffer<T, 1> BufferData(Data, NumData, {host_ptr});
    sycl::buffer<T, 1> BufferResults(nullptr, NumRes);
    sycl::nd_range<1> DataSz{GSZ, LSZ};
    BufferData.set_final_data(nullptr);

    auto &DeviceQueue = Queue();

    auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
      using LTy = sycl::accessor<T, 1, sycl_read_write, sycl_local>;
      LTy ReductionSums{sycl::range<1>{LSZ}, Cgh};
      auto Data = BufferData.template get_access<sycl_read>(Cgh);
      auto Results = BufferResults.template get_access<sycl_write>(Cgh);

      auto KernReduce = [=](sycl::nd_item<1> WorkItem) {
        const int N = WorkItem.get_global_id(0);
        const int L = WorkItem.get_local_id(0);
        const int Group = WorkItem.get_group(0);
        ReductionSums[L] = 0;
        for (int I = N; I < NumData; I += GSZ)
          ReductionSums[L] += Data[I];

        for (int Offset = LSZ / 2; Offset > 0; Offset /= 2) {
          WorkItem.barrier(sycl_local_fence);
          if (L < Offset)
            ReductionSums[L] += ReductionSums[L + Offset];
        }

        if (L == 0)
          Results[Group] = ReductionSums[0];
      };

      Cgh.parallel_for<class reduce_naive_buf<T>>(DataSz, KernReduce);
    });

    ProfInfo.emplace_back(Evt, "Calculating reduction");

    auto Res = BufferResults.template get_access<sycl_read>();

    // final combine
    Result = 0;
    for (int I = 0; I < NumRes; ++I)
      Result += Res[I];

    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycl::kernel_id kid = sycl::get_kernel_id<reduce_naive_buf<int>>();
  sycltesters::test_sequence<ReductionNaiveBuf<int>>(argc, argv, kid);
}
