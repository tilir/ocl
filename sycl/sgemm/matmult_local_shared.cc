//------------------------------------------------------------------------------
//
// Matrix multiplication with local memory kernel (SYCL vs serial CPU)
// Uses shared memory instead of buffers
//
// try: matmult_local_shared.exe -lsz=16
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
template <typename T> class mmult_local_shared;

using ConfigTy = sycltesters::sgemm::Config;

template <typename T>
class MatrixMultLocalShared : public sycltesters::MatrixMult<T> {
  using sycltesters::MatrixMult<T>::Queue;
  ConfigTy Cfg_;

public:
  MatrixMultLocalShared(sycl::queue &DeviceQueue, ConfigTy Cfg)
      : sycltesters::MatrixMult<T>(DeviceQueue), Cfg_(Cfg) {}

  sycltesters::EvtRet_t operator()(const T *Aptr, const T *Bptr, T *Cptr,
                                   size_t AX, size_t AY, size_t BY) override {
    assert(Aptr != nullptr && Bptr != nullptr && Cptr != nullptr);
    const auto LSZ = Cfg_.Lsz; // avoid implicit capture of this
    if ((AY % LSZ) != 0)
      throw std::runtime_error("Expect local size = multiple of AY");
    sycltesters::EvtVec_t ProfInfo;
    auto &DeviceQueue = Queue();

    auto *A = sycl::malloc_shared<T>(AX * AY, DeviceQueue);
    auto *B = sycl::malloc_shared<T>(AY * BY, DeviceQueue);
    auto *C = sycl::malloc_shared<T>(AX * BY, DeviceQueue);
    // alternative:
    // auto EvtCpyA = DeviceQueue.copy(Aptr, A, AX * AY);
    std::copy(Aptr, Aptr + AX * AY, A);
    std::copy(Bptr, Bptr + AY * BY, B);
    std::fill(Cptr, Cptr + AX * BY, 0);

    sycl::range<2> BlockSize{LSZ, LSZ};
    sycl::nd_range<2> Range{sycl::range<2>{AX, BY}, BlockSize};

    auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
      // local memory
      using LTy = sycl::accessor<T, 2, sycl_read_write, sycl_local>;
      LTy Asub{BlockSize, Cgh}, Bsub{BlockSize, Cgh};

      auto KernMul = [=](sycl::nd_item<2> It) {
        const int Row = It.get_local_id(0);
        const int Col = It.get_local_id(1);
        const int GlobalRow = LSZ * It.get_group(0) + Row;
        const int GlobalCol = LSZ * It.get_group(1) + Col;
        const int NumTiles = AY / LSZ;

        T Sum = 0;
        for (int Tile = 0; Tile < NumTiles; Tile++) {
          const int TiledRow = LSZ * Tile + Row;
          const int TiledCol = LSZ * Tile + Col;
          Asub[Row][Col] = A[GlobalRow * AY + TiledCol];
          Bsub[Row][Col] = B[TiledRow * BY + GlobalCol];
          // waiting for all threads to fill Asub[Row][Col]
          It.barrier(sycl_local_fence);
          for (int K = 0; K < LSZ; K++)
            Sum += Asub[Row][K] * Bsub[K][Col];
          // waiting for all threads to use Asub[Row][Col]
          It.barrier(sycl_local_fence);
        }
        C[GlobalRow * BY + GlobalCol] = Sum;
      };

      Cgh.parallel_for<class mmult_local_shared<T>>(Range, KernMul);
    });

    ProfInfo.push_back(Evt);
    Evt.wait();

    // copy back
    std::copy(C, C + AX * BY, Cptr);
    sycl::free(A, DeviceQueue);
    sycl::free(B, DeviceQueue);
    sycl::free(C, DeviceQueue);
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<MatrixMultLocalShared<float>>(argc, argv);
}