//------------------------------------------------------------------------------
//
// Vector addition, SYCL way, with explicit buffers
// no explicit sync required
//
//------------------------------------------------------------------------------
//
// This file is licensed after LGPL v3
// Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
//
//------------------------------------------------------------------------------

#include <CL/sycl.hpp>

#include <iostream>
#include <vector>

#include "testers.hpp"

// class is used for kernel name
template <typename T> class vector_add_buf;

template <typename T> class VectorAddBuf : public sycltesters::VectorAdd<T> {
  using sycltesters::VectorAdd<T>::Queue;

public:
  VectorAddBuf(cl::sycl::queue &DeviceQueue)
      : sycltesters::VectorAdd<T>(DeviceQueue) {}

  void operator()(T const *AVec, T const *BVec, T *CVec, size_t Sz) override {
    cl::sycl::range<1> NumOfItems{Sz};
    cl::sycl::buffer<T, 1> bufferA(AVec, NumOfItems);
    cl::sycl::buffer<T, 1> bufferB(BVec, NumOfItems);
    cl::sycl::buffer<T, 1> bufferC(CVec, NumOfItems);

    bufferA.set_final_data(nullptr);
    bufferB.set_final_data(nullptr);

    auto &DeviceQueue = Queue();

    DeviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto A = bufferA.template get_access<sycl_read>(cgh);
      auto B = bufferB.template get_access<sycl_read>(cgh);
      auto C = bufferC.template get_access<sycl_write>(cgh);

      auto kern = [A, B, C](cl::sycl::id<1> wiID) {
        C[wiID] = A[wiID] + B[wiID];
      };
      cgh.parallel_for<class vector_add_buf<T>>(NumOfItems, kern);
    });

    // host-side test that one vadd iteration is correct
#ifdef VERIFY
    auto A = bufferA.template get_access<sycl_read>();
    auto B = bufferB.template get_access<sycl_read>();
    auto C = bufferC.template get_access<sycl_read>();

    for (int i = 0; i < LIST_SIZE; ++i)
      if (C[i] != A[i] + B[i]) {
        std::cerr << "At index: " << i << ". ";
        std::cerr << C[i] << " != " << A[i] + B[i] << "\n";
        abort();
      }
#endif
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<VectorAddBuf<int>>(argc, argv);
}
