//------------------------------------------------------------------------------
//
// Vector addition example ESIMD version
//
//------------------------------------------------------------------------------
//
// This file is licensed after LGPL v3
// Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
//
//------------------------------------------------------------------------------

// to use block_load instead of simd::copy_to
#undef __SYCL_DEPRECATED
#define __SYCL_DEPRECATED(message)

#include <CL/sycl.hpp>
#include <ext/intel/experimental/esimd.hpp>

#include <iostream>
#include <vector>

#include "vadd_testers.hpp"

using namespace sycl::intel::experimental;

constexpr unsigned VL = 16;

template <typename T> class vector_add_esimd;

template <typename T> class VectorAddESIMD : public sycltesters::VectorAdd<T> {
  using sycltesters::VectorAdd<T>::Queue;

public:
  VectorAddESIMD(sycl::queue &DeviceQueue)
      : sycltesters::VectorAdd<T>(DeviceQueue) {}

  sycltesters::EvtRet_t operator()(T const *AVec, T const *BVec, T *CVec,
                                   size_t Sz) override {
    assert((Sz % VL) == 0 && "We need Sz to be evenly divisible by VL");
    sycltesters::EvtVec_t ProfInfo;
    sycl::buffer<T, 1> BufferA(AVec, Sz);
    sycl::buffer<T, 1> BufferB(BVec, Sz);
    sycl::buffer<T, 1> BufferC(CVec, Sz);

    BufferA.set_final_data(nullptr);
    BufferB.set_final_data(nullptr);

    auto &DeviceQueue = Queue();

    // SIMD-1 dispatch: we have all the GRF and vectorizing manually
    sycl::range<1> GlobalRange{Sz / VL};

    auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
      auto A = BufferA.template get_access<sycl_read>(Cgh);
      auto B = BufferB.template get_access<sycl_read>(Cgh);
      auto C = BufferC.template get_access<sycl_write>(Cgh);

      auto Kern = [=](sycl::id<1> Id) SYCL_ESIMD_KERNEL {
        unsigned Offset = Id * VL * sizeof(float);
        esimd::simd<T, VL> VA = esimd::block_load<T, VL>(A, Offset);
        esimd::simd<T, VL> VB = esimd::block_load<T, VL>(B, Offset);
        esimd::simd<T, VL> VC = VA + VB;
        esimd::block_store(C, Offset, VC);
      };

      Cgh.parallel_for<class vector_add_esimd<T>>(GlobalRange, Kern);
    });

    ProfInfo.push_back(Evt);

    // host-side test that one vadd iteration is correct
#ifdef VERIFY
    auto A = BufferA.template get_access<sycl_read>();
    auto B = BufferB.template get_access<sycl_read>();
    auto C = BufferC.template get_access<sycl_read>();

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
