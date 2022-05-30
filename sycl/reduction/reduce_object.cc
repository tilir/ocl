//------------------------------------------------------------------------------
//
// Reduction via reduction object (SYCL vs serial CPU).
// In this example SYCL uses special reduction object
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
template <typename T> class reduce_object_buf;

template <typename T>
class ReductionObjectBuf : public sycltesters::Reduction<T> {
  using sycltesters::Reduction<T>::Queue;
  using sycltesters::Reduction<T>::Bundle;
  ConfigTy Cfg_;

public:
  ReductionObjectBuf(sycl::queue &DeviceQueue, EBundleTy ExeBundle,
                     ConfigTy Cfg)
      : sycltesters::Reduction<T>(DeviceQueue, ExeBundle), Cfg_(Cfg) {}

  sycltesters::EvtRet_t operator()(const T *Data, size_t NumData,
                                   T &Result) override {
    assert(Data != nullptr && Bins != nullptr);
    sycltesters::EvtVec_t ProfInfo;
    Result = 0;

    sycl::buffer<T, 1> BufferData(Data, NumData);
    sycl::buffer<T, 1> BufferResult(&Result, 1);
    BufferData.set_final_data(nullptr);
    auto &DeviceQueue = Queue();

    auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
      auto Data = BufferData.template get_access<sycl_read>(Cgh);
      auto Result = BufferResult.template get_access<sycl_write>(Cgh);

      auto ReduceObj = sycl::reduction(BufferResult, Cgh, sycl::plus<>());
      auto KernReduce = [=](sycl::id<1> Idx, auto &Sum) { Sum += Data[Idx]; };
      Cgh.parallel_for<class reduce_object_buf<T>>(sycl::range<1>{NumData},
                                                   ReduceObj, KernReduce);
    });

    ProfInfo.emplace_back(Evt, "Calculating reduction");
    DeviceQueue.wait(); // or explicit host accessor to BufferResult
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
#ifdef BUG
  // not working!
  sycl::kernel_id kid = sycl::get_kernel_id<reduce_object_buf<int>>();
#else
  auto ids = sycl::get_kernel_ids();
  sycl::kernel_id kid = ids[0];
#endif

  sycltesters::test_sequence<ReductionObjectBuf<int>>(argc, argv, kid);
}
