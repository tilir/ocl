//------------------------------------------------------------------------------
//
// Matrix multiplication with MKL library (SYCL vs serial CPU)
//
// Macros to control things:
// -DMKLTRANS -- use transposed matrix
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
#include <mkl.h>
#include <oneapi/mkl/blas.hpp>

#include "sgemm_testers.hpp"

template <typename T>
class MatrixMultMKLShared : public sycltesters::MatrixMult<T> {
  using sycltesters::MatrixMult<T>::Queue;
  ConfigTy Cfg_;

public:
  MatrixMultMKLShared(sycl::queue &DeviceQueue, ConfigTy Cfg)
      : sycltesters::MatrixMult<T>(DeviceQueue), Cfg_(Cfg) {}

  sycltesters::EvtRet_t operator()(const T *Aptr, const T *Bptr, T *Cptr,
                                   size_t AX, size_t AY, size_t BY) override {
    assert(Aptr != nullptr && Bptr != nullptr && Cptr != nullptr);
    sycltesters::EvtVec_t ProfInfo;
    sycltesters::EvtVec_t GemmDependencies;
    auto &DeviceQueue = Queue();

    // C = alpha * op(A) * op(B)  + beta * C
    T Alpha = 1.0;
    T Beta = 0.0;

    oneapi::mkl::transpose TransA = oneapi::mkl::transpose::nontrans;
    oneapi::mkl::transpose TransB = oneapi::mkl::transpose::nontrans;

    auto *A = cl::sycl::malloc_shared<T>(AX * AY, DeviceQueue);
    auto *B = cl::sycl::malloc_shared<T>(AY * BY, DeviceQueue);
    auto *C = cl::sycl::malloc_shared<T>(AX * BY, DeviceQueue);
    std::copy(Aptr, Aptr + AX * AY, A);
#if !defined(MKLTRANS)
    std::copy(Bptr, Bptr + AY * BY, B);
#else
    TransB = oneapi::mkl::transpose::trans;
    for (int i = 0; i < AY; ++i)
      for (int j = 0; j < BY; ++j)
        B[j * AY + i] = Bptr[i * BY + j];
#endif
    std::copy(Cptr, Cptr + AX * BY, C); // zero-out

    auto Evt =
        oneapi::mkl::blas::gemm(DeviceQueue, TransA, TransB, AX, BY, AY, Alpha,
                                A, AX, B, AY, Beta, C, AX, GemmDependencies);

    ProfInfo.push_back(Evt);
    DeviceQueue.wait();

    // copy back
    std::copy(C, C + AX * BY, Cptr);
    cl::sycl::free(A, DeviceQueue);
    cl::sycl::free(B, DeviceQueue);
    cl::sycl::free(C, DeviceQueue);
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<MatrixMultMKLShared<float>>(argc, argv);
}