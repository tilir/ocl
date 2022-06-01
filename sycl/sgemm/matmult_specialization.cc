//------------------------------------------------------------------------------
//
// Matrix multiplication with simple kernel (SYCL vs serial CPU)
// illustrates usage of private constants
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
template <typename T> class mmult_specialized_buf;

// specialization constant for AY
const static sycl::specialization_id<int> AYC;

using ConfigTy = sycltesters::sgemm::Config;

template <typename T>
class MatrixMultSpecBuf : public sycltesters::MatrixMult<T> {
  using sycltesters::MatrixMult<T>::Queue;
  ConfigTy Cfg_;

public:
  MatrixMultSpecBuf(sycl::queue &DeviceQueue, ConfigTy Cfg)
      : sycltesters::MatrixMult<T>(DeviceQueue), Cfg_(Cfg) {}

  sycltesters::EvtRet_t operator()(const T *Aptr, const T *Bptr, T *Cptr,
                                   size_t AX, size_t AY, size_t BY) override {
    assert(Aptr != nullptr && Bptr != nullptr && Cptr != nullptr);
    sycltesters::EvtVec_t ProfInfo;
    sycl::range<2> Asz{AX, AY}, Bsz{AY, BY}, Csz{AX, BY};
    sycl::buffer<T, 2> BufA(Aptr, Asz), BufB(Bptr, Bsz), BufC(Cptr, Csz);
    BufA.set_final_data(nullptr);
    BufB.set_final_data(nullptr);

    auto &DeviceQueue = Queue();

    sycl::kernel_id KId = sycl::get_kernel_id<mmult_specialized_buf<T>>();
    sycl::kernel_bundle KbSrc =
        sycl::get_kernel_bundle<sycl::bundle_state::input>(
            DeviceQueue.get_context(), {KId});
    KbSrc.template set_specialization_constant<AYC>(AY);
    sycl::kernel_bundle Kb = sycl::build(KbSrc);

    auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
      auto A = BufA.template get_access<sycl_read>(Cgh);
      auto B = BufB.template get_access<sycl_read>(Cgh);
      auto C = BufC.template get_access<sycl_write>(Cgh);
      Cgh.use_kernel_bundle(Kb);

      auto Kernmul = [=](sycl::id<2> WorkItem, sycl::kernel_handler Kh) {
        const int Row = WorkItem.get(0);
        const int Col = WorkItem.get(1);
        const int AYK = Kh.template get_specialization_constant<AYC>();

        T Sum = 0;
        for (int K = 0; K < AYK; K++)
          Sum += A[Row][K] * B[K][Col];
        C[Row][Col] = Sum;
      };

      Cgh.parallel_for<class mmult_specialized_buf<T>>(Csz, Kernmul);
    });

    ProfInfo.push_back(Evt);
    DeviceQueue.wait(); // or explicit host accessor to BufC

    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<MatrixMultSpecBuf<float>>(argc, argv);
}