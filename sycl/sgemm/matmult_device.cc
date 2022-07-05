//------------------------------------------------------------------------------
//
// Matrix multiplication with simple kernel (SYCL vs serial CPU)
//
// Macros to control things:
// -DSHARED -- switch on shared memory instead of device
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

#include "sgemm_testers.hpp"

// class is used for kernel name
template <typename T> class mmult_naive_shared;

using ConfigTy = sycltesters::sgemm::Config;

template <typename T>
class MatrixMultNaiveShared : public sycltesters::MatrixMult<T> {
  using sycltesters::MatrixMult<T>::Queue;
  ConfigTy Cfg_;

public:
  MatrixMultNaiveShared(sycl::queue &DeviceQueue, ConfigTy Cfg)
      : sycltesters::MatrixMult<T>(DeviceQueue), Cfg_(Cfg) {}

  sycltesters::EvtRet_t operator()(const T *Aptr, const T *Bptr, T *Cptr,
                                   size_t AX, size_t AY, size_t BY) override {
    assert(Aptr != nullptr && Bptr != nullptr && Cptr != nullptr);
    sycltesters::EvtVec_t ProfInfo;
    sycl::range<2> Asz{AX, AY}, Bsz{AY, BY}, Csz{AX, BY};
    auto &DeviceQueue = Queue();
    int X = 1;

#ifdef SHARED
    auto *A = sycl::malloc_shared<T>(AX * AY, DeviceQueue);
    auto *B = sycl::malloc_shared<T>(AY * BY, DeviceQueue);
    auto *C = sycl::malloc_shared<T>(AX * BY, DeviceQueue);
    std::copy(Aptr, Aptr + AX * AY, A);
    std::copy(Bptr, Bptr + AY * BY, B);
#else
    auto *A = sycl::malloc_device<T>(AX * AY, DeviceQueue);
    auto *B = sycl::malloc_device<T>(AY * BY, DeviceQueue);
    auto *C = sycl::malloc_device<T>(AX * BY, DeviceQueue);
    auto EvtCpyA = DeviceQueue.copy(Aptr, A, AX * AY);
    auto EvtCpyB = DeviceQueue.copy(Bptr, B, AY * BY);
    ProfInfo.emplace_back(EvtCpyA, "Copy A forth");
    ProfInfo.emplace_back(EvtCpyB, "Copy B forth");
#endif

    auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
#ifndef SHARED
      Cgh.depends_on(EvtCpyA);
      Cgh.depends_on(EvtCpyB);
#endif

      auto Kernmul = [=](sycl::id<2> WorkItem) {
        const int Row = WorkItem.get(0);
        const int Col = WorkItem.get(1);

        T Sum = 0;
        for (int K = 0; K < AY; K++)
          Sum += A[X * Row * AY + K] * B[K * BY + X * Col];
        C[Row * BY + X * Col] = Sum;
      };

      Cgh.parallel_for<class mmult_naive_shared<T>>(Csz, Kernmul);
    });

    ProfInfo.emplace_back(Evt, "Main calculation");

#ifdef SHARED
    DeviceQueue.wait();
    std::copy(C, C + AX * BY, Cptr);
#else
    auto EvtCpyC = DeviceQueue.copy(C, Cptr, AX * BY, Evt);
    ProfInfo.emplace_back(EvtCpyC, "Copy C back");
    DeviceQueue.wait();
#endif

    sycl::free(A, DeviceQueue);
    sycl::free(B, DeviceQueue);
    sycl::free(C, DeviceQueue);
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<MatrixMultNaiveShared<float>>(argc, argv);
}