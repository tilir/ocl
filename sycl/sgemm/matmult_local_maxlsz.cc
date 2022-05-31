//------------------------------------------------------------------------------
//
// Matrix multiplication with local memory kernel (SYCL vs serial CPU)
// Extends global iteration space to have uniform subgroup of maximum size.
//
// No visible effect on TGLLP because max WG size 256 means max LSZ is 16
// Interesting hang with:
// > matmult_local_maxlsz.exe -bsz=1 -ax=101 -ay=102 -by=103
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
template <typename T> class mmult_local_buf;

using ConfigTy = sycltesters::sgemm::Config;

template <typename T>
class MatrixMultLocalBuf : public sycltesters::MatrixMult<T> {
  using sycltesters::MatrixMult<T>::Queue;
  ConfigTy Cfg_;

  // roundup(4, 2) == 4
  // roundup(4, 3) == 6
  // roundup(5, 2) == 6
  int roundup(int n, int m) {
    if ((n % m) == 0)
      return n;
    return ((n / m) + 1) * m;
  }

  int isqrt(int n) { return sqrt(n); }

public:
  MatrixMultLocalBuf(sycl::queue &DeviceQueue, ConfigTy Cfg)
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
    const int MLSZ =
        DeviceQueue.get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    const int LSZ = isqrt(MLSZ);

    // in this case we need to account for tail
    const int NumTiles = AY / LSZ;
    const int TailSize = AY - NumTiles * LSZ;
    const int AXR = roundup(AX, LSZ);
    const int BYR = roundup(BY, LSZ);
    sycltesters::qout << "MAX WG size = " << MLSZ << "\n";
    sycltesters::qout << "Selected local size dimension = " << LSZ << "\n";
    sycltesters::qout << "# tiles = " << NumTiles << "\n";
    sycltesters::qout << "Tail size = " << TailSize << "\n";

    sycl::range<2> Sizes{AXR, BYR};
    sycl::range<2> BlockSize{LSZ, LSZ};
    sycl::nd_range<2> Range{Sizes, BlockSize};

    auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
      auto A = BufA.template get_access<sycl_read>(Cgh);
      auto B = BufB.template get_access<sycl_read>(Cgh);
      auto C = BufC.template get_access<sycl_write>(Cgh);

      // local memory
      using LTy = sycl::accessor<T, 2, sycl_read_write, sycl_local>;
      LTy Asub{BlockSize, Cgh}, Bsub{BlockSize, Cgh};

      auto KernMul = [=](sycl::nd_item<2> It) {
        const int Row = It.get_local_id(0);
        const int Col = It.get_local_id(1);
        const int GlobalRow = It.get_global_id(0);
        const int GlobalCol = It.get_global_id(1);

        // ignore padding threads
        if (GlobalRow >= AX || GlobalCol >= BY)
          return;

        T Sum = 0;
        for (int Tile = 0; Tile < NumTiles; Tile++) {
          const int TiledRow = LSZ * Tile + Row;
          const int TiledCol = LSZ * Tile + Col;
          Asub[Row][Col] = A[GlobalRow][TiledCol];
          Bsub[Row][Col] = B[TiledRow][GlobalCol];
          It.barrier(sycl_local_fence);

          for (int K = 0; K < LSZ; K++)
            Sum += Asub[Row][K] * Bsub[K][Col];
          It.barrier(sycl_local_fence);
        }

        // tail because max local size almost for sure
        // will not be a multiple of AY
        for (int K = 0; K < TailSize; K++) {
          const int TailId = NumTiles * LSZ + K;
          Sum += A[Row][TailId] * B[TailId][Col];
        }

        C[GlobalRow][GlobalCol] = Sum;
      };

      Cgh.parallel_for<class mmult_local_buf<T>>(Range, KernMul);
    });

    ProfInfo.emplace_back(Evt, "Main execution");
    DeviceQueue.wait(); // or explicit host accessor to BufC
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<MatrixMultLocalBuf<float>>(argc, argv);
}