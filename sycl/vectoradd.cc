//------------------------------------------------------------------------------
//
// Vector addition, SYCL way, with explicit buffers
// no explicit sync required
//
// Windows OneAPI console
// build: dpcpp -EHsc -std=c++20 -fsycl vectoradd.cc
// run: vectoradd.exe -size=20000 -nreps=2000
//
//------------------------------------------------------------------------------
//
// This file is licensed after LGPL v3
// Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
//
//------------------------------------------------------------------------------

#include <iostream>
#include <vector>

#include <CL/sycl.hpp>

#include "vadd_testers.hpp"

// class is used for kernel name
template <typename T> class vector_add_buf;

template <typename T> class VectorAddBuf : public sycltesters::VectorAdd<T> {
  using sycltesters::VectorAdd<T>::Queue;

public:
  VectorAddBuf(cl::sycl::queue &DeviceQueue)
      : sycltesters::VectorAdd<T>(DeviceQueue) {}

  sycltesters::EvtRet_t operator()(T const *AVec, T const *BVec, T *CVec,
                                   size_t Sz) override {
    std::vector<cl::sycl::event> ProfInfo;
    cl::sycl::range<1> NumOfItems{Sz};
    cl::sycl::buffer<T, 1> bufferA(AVec, NumOfItems, {host_ptr});
    cl::sycl::buffer<T, 1> bufferB(BVec, NumOfItems, {host_ptr});
    cl::sycl::buffer<T, 1> bufferC(CVec, NumOfItems, {host_ptr});

    bufferA.set_final_data(nullptr);
    bufferB.set_final_data(nullptr);

    auto &DeviceQueue = Queue();

    auto Evt = DeviceQueue.submit([&](cl::sycl::handler &Cgh) {
      auto A = bufferA.template get_access<sycl_read>(Cgh);
      auto B = bufferB.template get_access<sycl_read>(Cgh);
      auto C = bufferC.template get_access<sycl_write>(Cgh);

      auto kern = [A, B, C](cl::sycl::id<1> wiID) {
        C[wiID] = A[wiID] + B[wiID];
      };
      Cgh.parallel_for<class vector_add_buf<T>>(NumOfItems, kern);
    });

    ProfInfo.push_back(Evt);
    Evt.wait();

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
  sycltesters::test_sequence<VectorAddBuf<int>>(argc, argv);
}
