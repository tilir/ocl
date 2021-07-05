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

  sycltesters::EvtRet_t operator()(T const *AVec, T const *BVec, T *CVec,
                                   size_t Sz) override {
    std::vector<cl::sycl::event> ProfInfo;
    auto &DeviceQueue = Queue();
    int *A = cl::sycl::malloc_device<T>(Sz, DeviceQueue);
    int *B = cl::sycl::malloc_device<T>(Sz, DeviceQueue);
    int *C = cl::sycl::malloc_device<T>(Sz, DeviceQueue);

    // kernels to copy to device
    auto eA = DeviceQueue.submit(
        [&](cl::sycl::handler &cgh) { cgh.memcpy(A, AVec, Sz * sizeof(T)); });
    ProfInfo.push_back(eA);

    auto eB = DeviceQueue.submit(
        [&](cl::sycl::handler &cgh) { cgh.memcpy(B, BVec, Sz * sizeof(T)); });
    ProfInfo.push_back(eB);

    // vector addition
    cl::sycl::range<1> numOfItems{Sz};
    auto eC = DeviceQueue.submit([&](cl::sycl::handler &cgh) {
      cgh.depends_on({eA, eB});
      auto kern = [A, B, C](cl::sycl::id<1> wiID) {
        C[wiID] = A[wiID] + B[wiID];
      };
      cgh.parallel_for<class vector_add_device<T>>(numOfItems, kern);
    });
    ProfInfo.push_back(eC);

    // copy back
    DeviceQueue.submit([&](cl::sycl::handler &cgh) {
      cgh.depends_on(eC);
      cgh.memcpy(CVec, C, Sz * sizeof(T));
    });

    // last wait inevitable
    DeviceQueue.wait();

// host-side test that one vadd iteration is correct
#ifdef VERIFY
    for (int i = 0; i < Sz; ++i)
      if (CVec[i] != AVec[i] + BVec[i]) {
        std::cerr << "At index: " << i << ". ";
        std::cerr << CVec[i] << " != " << AVec[i] + BVec[i] << "\n";
        abort();
      }
#endif
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<VectorAddBuf<int>>(argc, argv);
}
