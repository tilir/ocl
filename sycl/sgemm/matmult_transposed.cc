//------------------------------------------------------------------------------
//
// Matrix multiplication with simple kernel (SYCL vs serial CPU)
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
template <typename T> class mmult_shared_transposed;

using ConfigTy = sycltesters::sgemm::Config;

template <typename T>
class MatrixMultSharedTransposed : public sycltesters::MatrixMult<T> {
  using sycltesters::MatrixMult<T>::Queue;
  ConfigTy Cfg_;

public:
  MatrixMultSharedTransposed(sycl::queue &DeviceQueue, ConfigTy Cfg)
      : sycltesters::MatrixMult<T>(DeviceQueue), Cfg_(Cfg) {}

  sycltesters::EvtRet_t operator()(const T *Aptr, const T *Bptr, T *Cptr,
                                   size_t AX, size_t AY, size_t BY) override {
    assert(Aptr != nullptr && Bptr != nullptr && Cptr != nullptr);
    sycltesters::EvtVec_t ProfInfo;
    sycl::range<2> Asz{AX, AY}, Bsz{AY, BY}, Csz{AX, BY};

    auto &DeviceQueue = Queue();

    auto *A = sycl::malloc_shared<T>(AX * AY, DeviceQueue);
    auto *B = sycl::malloc_shared<T>(AY * BY, DeviceQueue);
    auto *C = sycl::malloc_shared<T>(AX * BY, DeviceQueue);
    std::copy(Aptr, Aptr + AX * AY, A);
    // transpose matrix B
    for (int i = 0; i < AY; ++i)
      for (int j = 0; j < BY; ++j)
        B[j * AY + i] = Bptr[i * BY + j];
    std::fill(Cptr, Cptr + AX * BY, 0);

    auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
      auto Kernmul = [=](sycl::id<2> WorkItem) {
        const int Row = WorkItem.get(0);
        const int Col = WorkItem.get(1);
        T Sum = 0;
        // iterate the same K dimension
        for (int K = 0; K < AY; K++)
          Sum += A[Row * AY + K] * B[Col * AY + K];
        C[Row * BY + Col] = Sum;
      };

      Cgh.parallel_for<class mmult_shared_transposed<T>>(Csz, Kernmul);
    });

    ProfInfo.push_back(Evt);
    DeviceQueue.wait();
    std::copy(C, C + AX * BY, Cptr);
    sycl::free(A, DeviceQueue);
    sycl::free(B, DeviceQueue);
    sycl::free(C, DeviceQueue);
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<MatrixMultSharedTransposed<float>>(argc, argv);
}