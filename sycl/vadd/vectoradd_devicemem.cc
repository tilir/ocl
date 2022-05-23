//------------------------------------------------------------------------------
//
// Vector addition, SYCL way, with malloc_device.
// We may set depends explicitly, so no explicit wait here, except last.
// illustrates cgh.memcpy and cgh.depends_on usage.
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
    sycltesters::EvtVec_t ProfInfo;
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

    // vector addition
    cl::sycl::range<1> numOfItems{Sz};
    auto EvtC = DeviceQueue.submit([&](cl::sycl::handler &cgh) {
      cgh.depends_on({EvtA, EvtB});
      auto kern = [A, B, C](cl::sycl::id<1> wiID) {
        C[wiID] = A[wiID] + B[wiID];
      };
      cgh.parallel_for<class vector_add_device<T>>(numOfItems, kern);
    });
    ProfInfo.push_back(EvtC);

    // copy back
    auto EvtD = DeviceQueue.submit([&](cl::sycl::handler &cgh) {
      cgh.depends_on(EvtC);
      cgh.memcpy(CVec, C, Sz * sizeof(T));
    });
    ProfInfo.push_back(EvtD);

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

    cl::sycl::free(A, DeviceQueue);
    cl::sycl::free(B, DeviceQueue);
    cl::sycl::free(C, DeviceQueue);
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<VectorAddDevice<int>>(argc, argv);
}
