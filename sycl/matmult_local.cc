//------------------------------------------------------------------------------
//
// Matrix multiplication with local memory kernel (SYCL vs serial CPU)
//
// try: matmult_local.exe -lsz=16
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

template <typename T>
class MatrixMultLocalBuf : public sycltesters::MatrixMult<T> {
  using sycltesters::MatrixMult<T>::Queue;
  unsigned Lsz_;

public:
  MatrixMultLocalBuf(cl::sycl::queue &DeviceQueue, unsigned Lsz)
      : sycltesters::MatrixMult<T>(DeviceQueue), Lsz_(Lsz) {}

  sycltesters::EvtRet_t operator()(const T *Aptr, const T *Bptr, T *Cptr,
                                   size_t AX, size_t AY, size_t BY) override {
    assert(Aptr != nullptr && Bptr != nullptr && Cptr != nullptr);
    const auto LSZ = Lsz_; // avoid implicit capture of this
    assert((AY % LSZ) == 0);
    std::vector<cl::sycl::event> ProfInfo;
    cl::sycl::range<2> Asz{AX, AY}, Bsz{AY, BY}, Csz{AX, BY};
    cl::sycl::buffer<T, 2> BufferA(Aptr, Asz), BufferB(Bptr, Bsz),
        BufferC(Cptr, Csz);

    BufferA.set_final_data(nullptr);
    BufferB.set_final_data(nullptr);

    auto &DeviceQueue = Queue();

    cl::sycl::range<2> BlockSize{LSZ, LSZ};
    cl::sycl::nd_range<2> Range{Csz, BlockSize};

    auto Evt = DeviceQueue.submit([&](cl::sycl::handler &Cgh) {
      auto A = BufferA.template get_access<sycl_read>(Cgh);
      auto B = BufferB.template get_access<sycl_read>(Cgh);
      auto C = BufferC.template get_access<sycl_write>(Cgh);

      // local memory
      cl::sycl::accessor<T, 2, sycl_read_write, sycl_local> Asub(BlockSize,
                                                                 Cgh);
      cl::sycl::accessor<T, 2, sycl_read_write, sycl_local> Bsub(BlockSize,
                                                                 Cgh);

      auto KernMul = [=](cl::sycl::nd_item<2> It) {
        int Row = It.get_local_id(0);
        int Col = It.get_local_id(1);
        int GlobalRow = LSZ * It.get_group(0) + Row;
        int GlobalCol = LSZ * It.get_group(1) + Col;
        int NumTiles = AY / LSZ;

        T Sum = 0;

        for (int Tile = 0; Tile < NumTiles; Tile++) {
          int TiledRow = LSZ * Tile + Row;
          int TiledCol = LSZ * Tile + Col;
          Asub[Row][Col] = A[GlobalRow][TiledCol];
          Bsub[Row][Col] = B[TiledRow][GlobalCol];
#ifndef NOBARRIER
          It.barrier(sycl_local_fence);
#endif

          for (int K = 0; K < LSZ; K++)
            Sum += Asub[Row][K] * Bsub[K][Col];
#ifndef NOBARRIER
          It.barrier(sycl_local_fence);
#endif
        }
        C[GlobalRow][GlobalCol] = Sum;
      };

      Cgh.parallel_for<class mmult_local_buf<T>>(Range, KernMul);
    });

    ProfInfo.push_back(Evt);
    Evt.wait();

    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<MatrixMultLocalBuf<float>>(argc, argv);
}