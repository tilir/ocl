//==---------------- vadd_1d.cpp  - DPC++ ESIMD on-device test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// to use block_load instead of simd::copy_to
#undef __SYCL_DEPRECATED
#define __SYCL_DEPRECATED(message)

#include <CL/sycl.hpp>
#include <ext/intel/experimental/esimd.hpp>

#include <iostream>
#include <vector>

#include "vadd_testers.hpp"

using namespace cl::sycl::intel::experimental;

constexpr unsigned VL = 16;

// class is used for kernel name
template <typename T> class vector_add_esimd;

template <typename T> class VectorAddESIMD : public sycltesters::VectorAdd<T> {
  using sycltesters::VectorAdd<T>::Queue;

public:
  VectorAddESIMD(cl::sycl::queue &DeviceQueue)
      : sycltesters::VectorAdd<T>(DeviceQueue) {}

  sycltesters::EvtRet_t operator()(T const *AVec, T const *BVec, T *CVec,
                                   size_t Sz) override {
    assert((Sz % VL) == 0 && "We need Sz to be evenly divisible by VL");
    sycltesters::EvtVec_t ProfInfo;
    cl::sycl::buffer<T, 1> bufferA(AVec, Sz, {host_ptr});
    cl::sycl::buffer<T, 1> bufferB(BVec, Sz, {host_ptr});
    cl::sycl::buffer<T, 1> bufferC(CVec, Sz, {host_ptr});

    bufferA.set_final_data(nullptr);
    bufferB.set_final_data(nullptr);

    auto &DeviceQueue = Queue();

    // We need that many workgroups
    cl::sycl::range<1> GlobalRange{Sz / VL};

    auto Evt = DeviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto A = bufferA.template get_access<sycl_read>(cgh);
      auto B = bufferB.template get_access<sycl_read>(cgh);
      auto C = bufferC.template get_access<sycl_write>(cgh);

      auto kern = [A, B, C](cl::sycl::id<1> wiID) SYCL_ESIMD_KERNEL {
        unsigned Offset = wiID * VL * sizeof(float);
        esimd::simd<T, VL> VA = esimd::block_load<T, VL>(A, Offset);
        esimd::simd<T, VL> VB = esimd::block_load<T, VL>(B, Offset);
        esimd::simd<T, VL> VC = VA + VB;
        esimd::block_store(C, Offset, VC);
      };

      cgh.parallel_for<class vector_add_esimd<T>>(GlobalRange, kern);
    });

    ProfInfo.push_back(Evt);

    // host-side test that one vadd iteration is correct
#ifdef VERIFY
    auto A = bufferA.template get_access<sycl_read>();
    auto B = bufferB.template get_access<sycl_read>();
    auto C = bufferC.template get_access<sycl_read>();

    for (int i = 0; i < Sz; ++i)
      if (C[i] != A[i] + B[i]) {
        std::cerr << "At index: " << i << ". ";
        std::cerr << C[i] << " != " << A[i] + B[i] << "\n";
        abort();
      }
#endif
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<VectorAddESIMD<int>>(argc, argv);
}
