//------------------------------------------------------------------------------
//
// Vector addition, SYCL way, with malloc_device
// We may set depends explicitly, so no explicit wait here, except last
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
template <typename T> class vector_add_device;

template <typename T> class VectorAddBuf : public sycltesters::VectorAdd<T> {
  using sycltesters::VectorAdd<T>::Queue;

public:
  VectorAddBuf(cl::sycl::queue &DeviceQueue)
      : sycltesters::VectorAdd<T>(DeviceQueue) {}

  void operator()(T const *AVec, T const *BVec, T *CVec, size_t Sz) override {

    int *A = cl::sycl::malloc_device<T>(sz, deviceQueue);
    int *B = cl::sycl::malloc_device<T>(sz, deviceQueue);
    int *C = cl::sycl::malloc_device<T>(sz, deviceQueue);

    // kernels to copy to device
    auto eA = deviceQueue.submit(
        [&](cl::sycl::handler &cgh) { cgh.memcpy(A, AVec, Sz * sizeof(T)); });

    auto eB = deviceQueue.submit(
        [&](cl::sycl::handler &cgh) { cgh.memcpy(B, BVec, Sz * sizeof(T)); });

    // vector addition
    cl::sycl::range<1> numOfItems{sz};
    auto eC = deviceQueue.submit([&](cl::sycl::handler &cgh) {
      cgh.depends_on({eA, eB});
      auto kern = [A, B, C](cl::sycl::id<1> wiID) {
        C[wiID] = A[wiID] + B[wiID];
      };
      cgh.parallel_for<class vector_add_device<T>>(numOfItems, kern);
    });

    // copy back
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      cgh.depends_on(eC);
      cgh.memcpy(CVec, C, Sz * sizeof(T));
    });

    // last wait inevitable
    deviceQueue.wait();

// host-side test that one vadd iteration is correct
#ifdef VERIFY
    for (int i = 0; i < LIST_SIZE; ++i)
      if (CVec[i] != Avec[i] + Bvec[i]) {
        std::cerr << "At index: " << i << ". ";
        std::cerr << pc[i] << " != " << pa[i] + pb[i] << "\n";
        abort();
      }
#endif
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<VectorAddBuf<int>>(argc, argv);
}
