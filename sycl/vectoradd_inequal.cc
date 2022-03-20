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

#include "vadd_testers.hpp"

// class is used for kernel name
template <typename T> class vector_add_shared;

template <typename T> class VectorAddShared : public sycltesters::VectorAdd<T> {
  using sycltesters::VectorAdd<T>::Queue;

public:
  VectorAddShared(cl::sycl::queue &DeviceQueue)
      : sycltesters::VectorAdd<T>(DeviceQueue) {}

  sycltesters::EvtRet_t operator()(T const *AVec, T const *BVec, T *CVec,
                                   size_t Sz) override {
    std::vector<cl::sycl::event> ProfInfo;
    auto &DeviceQueue = Queue();

#ifdef HOST_ALLOC
    int *A = cl::sycl::malloc_host<T>(Sz, DeviceQueue);
    int *B = cl::sycl::malloc_host<T>(Sz, DeviceQueue);
    int *C = cl::sycl::malloc_host<T>(Sz, DeviceQueue);
#else
    int *A = cl::sycl::malloc_shared<T>(Sz, DeviceQueue);
    int *B = cl::sycl::malloc_shared<T>(Sz, DeviceQueue);
    int *C = cl::sycl::malloc_shared<T>(Sz, DeviceQueue);
#endif

    // this multiplier is intended to break stateless-to-statefull
    int *Mult = cl::sycl::malloc_shared<int>(1, DeviceQueue);
    *Mult = 1;

    std::copy(AVec, AVec + Sz, A);
    std::copy(BVec, BVec + Sz, B);

    // vector addition
    cl::sycl::range<1> numOfItems{Sz};

    // requires -fsycl-unnamed-lambda option to be added
    auto Evt = DeviceQueue.parallel_for(
        numOfItems, [=](auto n) { C[*Mult * n] = A[n] + B[n]; });
    ProfInfo.push_back(Evt);

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
  sycltesters::test_sequence<VectorAddShared<int>>(argc, argv);
}
