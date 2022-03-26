//------------------------------------------------------------------------------
//
// Vector addition, SYCL way, with malloc_device
// Explicit wait on queue is basic synchronization mechanism
// Try -DNOWAIT to trigger strange bugs
// Try inorder queues as an alternative
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

#include "vadd_testers.hpp"

// class is used for kernel name
template <typename T> class vector_add_device;

template <typename T> class VectorAddDevice : public sycltesters::VectorAdd<T> {
  using sycltesters::VectorAdd<T>::Queue;

public:
  VectorAddDevice(cl::sycl::queue &DeviceQueue)
      : sycltesters::VectorAdd<T>(DeviceQueue) {}

  sycltesters::EvtRet_t operator()(T const *AVec, T const *BVec, T *CVec,
                                   size_t Sz) override {
    std::vector<cl::sycl::event> ProfInfo;
    auto &DeviceQueue = Queue();
    int *A = cl::sycl::malloc_device<T>(Sz, DeviceQueue);
    int *B = cl::sycl::malloc_device<T>(Sz, DeviceQueue);
    int *C = cl::sycl::malloc_device<T>(Sz, DeviceQueue);

    // kernels to copy to device
    auto EvtA = DeviceQueue.submit(
        [&](cl::sycl::handler &cgh) { cgh.memcpy(A, AVec, Sz * sizeof(T)); });
    ProfInfo.push_back(EvtA);
    auto EvtB = DeviceQueue.submit(
        [&](cl::sycl::handler &cgh) { cgh.memcpy(B, BVec, Sz * sizeof(T)); });
    ProfInfo.push_back(EvtB);
#ifndef NOWAIT
    DeviceQueue.wait();
#endif

    // vector addition
    cl::sycl::range<1> numOfItems{Sz};
    auto EvtC = DeviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto kern = [A, B, C](cl::sycl::id<1> wiID) {
        C[wiID] = A[wiID] + B[wiID];
      };
      cgh.parallel_for<class vector_add_device<T>>(numOfItems, kern);
    });
    ProfInfo.push_back(EvtC);
#ifndef NOWAIT
    DeviceQueue.wait();
#endif

    // copy back
    auto EvtD = DeviceQueue.submit(
        [&](cl::sycl::handler &cgh) { cgh.memcpy(CVec, C, Sz * sizeof(T)); });
    ProfInfo.push_back(EvtD);
#ifndef NOWAIT
    DeviceQueue.wait();
#endif

    // trying to access device memory on host
#ifdef TRYACC
    std::cout << C[0] << std::endl;
#endif

    // host-side test that one vadd iteration is correct
#ifdef VERIFY
    for (int i = 0; i < Sz; ++i)
      if (CVec[i] != AVec[i] + BVec[i]) {
        std::cerr << "At index: " << i << ". ";
        std::cerr << CVec[i] << " != " << AVec[i] + BVec[i] << "\n";
        abort();
      }
#endif
    cl::sycl::free(A, DeviceQueue);
    cl::sycl::free(B, DeviceQueue);
    cl::sycl::free(C, DeviceQueue);
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<VectorAddDevice<int>>(argc, argv);
}
