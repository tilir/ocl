//------------------------------------------------------------------------------
//
// Matrix multiplication with simple kernel (SYCL vs serial CPU)
//
// Macros to control things:
// -DUSM_ALLOC -- switch on USM allocation
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
template <typename T> class mmult_naive_system;

using ConfigTy = sycltesters::sgemm::Config;

template <typename T>
class MatrixMultNaiveSystem : public sycltesters::MatrixMult<T> {
  using sycltesters::MatrixMult<T>::Queue;
  ConfigTy Cfg_;

public:
  MatrixMultNaiveSystem(sycl::queue &DeviceQueue, ConfigTy Cfg)
      : sycltesters::MatrixMult<T>(DeviceQueue), Cfg_(Cfg) {}

  sycltesters::EvtRet_t operator()(const T *Aptr, const T *Bptr, T *Cptr,
                                   size_t AX, size_t AY, size_t BY) override {
    assert(Aptr != nullptr && Bptr != nullptr && Cptr != nullptr);
    sycltesters::EvtVec_t ProfInfo;
    sycl::range<2> Asz{AX, AY}, Bsz{AY, BY}, Csz{AX, BY};
    auto &DeviceQueue = Queue();
    auto D = DeviceQueue.get_device();

#ifdef USM_ALLOC
    auto ShAlloc = D.has(sycl::aspect::usm_shared_allocations);
    if (!ShAlloc)
      throw std::runtime_error("Shared allocations support required");

    using AllocTy = sycl::usm_allocator<int, sycl::usm::alloc::shared>;
    AllocTy Alloc{DeviceQueue};
    std::vector<int, AllocTy> AData(AX * AY, Alloc);
    std::vector<int, AllocTy> BData(AY * BY, Alloc);
    std::vector<int, AllocTy> CData(AX * BY, Alloc);
    auto *A = AData.data();
    auto *B = BData.data();
    auto *C = CData.data();

    std::copy(Aptr, Aptr + AX * AY, A);
    std::copy(Bptr, Bptr + AY * BY, B);
#else
    // in theory USM system allocations shall work fine if aspect on
    // in practice I have a bug here
    auto SysAlloc = D.has(sycl::aspect::usm_system_allocations);
    if (!SysAlloc)
      throw std::runtime_error("System allocations support required");

    auto *A = Aptr;
    auto *B = Bptr;
    auto *C = Cptr;
#endif

    auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
      auto Kernmul = [=](sycl::id<2> WorkItem) {
        const int Row = WorkItem.get(0);
        const int Col = WorkItem.get(1);

        T Sum = 0;
        for (int K = 0; K < AY; K++)
          Sum += A[Row * AY + K] * B[K * BY + Col];
        C[Row * BY + Col] = Sum;
      };

      Cgh.parallel_for<class mmult_naive_system<T>>(Csz, Kernmul);
    });

    DeviceQueue.wait();
#ifdef USM_ALLOC
    std::copy(C, C + AX * BY, Cptr);
#endif

    ProfInfo.emplace_back(Evt, "Main calculation");
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<MatrixMultNaiveSystem<float>>(argc, argv);
}