//------------------------------------------------------------------------------
//
// Matrix multiplication with simple kernel (SYCL vs serial CPU)
//
// Macros to control things:
// -DNONTRANSPOSED -- switch off transposition
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

const static sycl::specialization_id<int> AYC;
const static sycl::specialization_id<int> BYC;

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

    auto *A = cl::sycl::malloc_shared<T>(AX * AY, DeviceQueue);
    auto *B = cl::sycl::malloc_shared<T>(AY * BY, DeviceQueue);
    auto *C = cl::sycl::malloc_shared<T>(AX * BY, DeviceQueue);
    std::copy(Aptr, Aptr + AX * AY, A);
    std::copy(Bptr, Bptr + AY * BY, B);
    std::fill(Cptr, Cptr + AX * BY, 0);

    auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
      Cgh.template set_specialization_constant<AYC>(AY);
      Cgh.template set_specialization_constant<BYC>(BY);
      auto Kernmul = [=](sycl::id<2> WorkItem, sycl::kernel_handler Kh) {
        const int Row = WorkItem.get(0);
        const int Col = WorkItem.get(1);
        const int AYK = Kh.template get_specialization_constant<AYC>();
        const int BYK = Kh.template get_specialization_constant<BYC>();

        T Sum = 0;
        for (int K = 0; K < AYK; K++)
          Sum += A[Row * AYK + K] * B[K * BYK + Col];
        C[Row * BYK + Col] = Sum;
      };

      Cgh.parallel_for<class mmult_shared_transposed<T>>(Csz, Kernmul);
    });

    ProfInfo.push_back(Evt);
    DeviceQueue.wait();
    std::copy(C, C + AX * BY, Cptr);
    cl::sycl::free(A, DeviceQueue);
    cl::sycl::free(B, DeviceQueue);
    cl::sycl::free(C, DeviceQueue);

    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<MatrixMultSharedTransposed<float>>(argc, argv);
}