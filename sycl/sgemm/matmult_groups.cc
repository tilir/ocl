//------------------------------------------------------------------------------
//
// Matrix multiplication with local memory kernel (SYCL vs serial CPU)
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
template <typename T> class mmult_groups;

template <typename T>
class MatrixMultGroups : public sycltesters::MatrixMult<T> {
  using sycltesters::MatrixMult<T>::Queue;
  unsigned Lsz_;

public:
  MatrixMultGroups(cl::sycl::queue &DeviceQueue, unsigned Lsz)
      : sycltesters::MatrixMult<T>(DeviceQueue), Lsz_(Lsz) {}

  sycltesters::EvtRet_t operator()(const T *Aptr, const T *Bptr, T *Cptr,
                                   size_t AX, size_t AY, size_t BY) override {
    const auto LSZ = Lsz_; // avoid implicit capture of this
    assert(Aptr != nullptr && Bptr != nullptr && Cptr != nullptr);
    assert((AY % LSZ) == 0 && (AY % LSZ) == 0 && (BY % LSZ) == 0);
    std::vector<cl::sycl::event> ProfInfo;
    cl::sycl::range<2> Asz{AX, AY}, Bsz{AY, BY}, Csz{AX, BY};
    cl::sycl::buffer<T, 2> BufferA(Aptr, Asz), BufferB(Bptr, Bsz),
        BufferC(Cptr, Csz);

    BufferA.set_final_data(nullptr);
    BufferB.set_final_data(nullptr);

    auto &DeviceQueue = Queue();
    auto Evt = DeviceQueue.submit([&](cl::sycl::handler &Cgh) {
      auto A = BufferA.template get_access<sycl_read>(Cgh);
      auto B = BufferB.template get_access<sycl_read>(Cgh);
      auto C = BufferC.template get_access<sycl_write>(Cgh);

      cl::sycl::range<2> NumGroups{AX / LSZ, BY / LSZ};
      cl::sycl::range<2> BlockSize{LSZ, LSZ};

      // local memory
      using LTy = cl::sycl::accessor<T, 2, sycl_read_write, sycl_local>;
      LTy Asub(BlockSize, Cgh);
      LTy Bsub(BlockSize, Cgh);

      auto KernMul = [=](cl::sycl::group<2> Group) {
        const int GroupRow = Group.get_id(0);
        const int GroupCol = Group.get_id(1);
        const int NumTiles = AY / LSZ;

        Group.parallel_for_work_item([&](cl::sycl::h_item<2> It) {
          int GlobalRow = It.get_global_id()[0];
          int GlobalCol = It.get_global_id()[1];
          C[GlobalRow][GlobalCol] = 0;
        });

        for (int Tile = 0; Tile < NumTiles; Tile++) {
          Group.parallel_for_work_item([&](cl::sycl::h_item<2> It) {
            int GlobalRow = It.get_global_id()[0];
            int GlobalCol = It.get_global_id()[1];
            int Row = It.get_local_id()[0];
            int Col = It.get_local_id()[1];
            int TiledRow = LSZ * Tile + Row;
            int TiledCol = LSZ * Tile + Col;

            Asub[Row][Col] = A[GlobalRow][TiledCol];
            Bsub[Row][Col] = B[TiledRow][GlobalCol];
          });

          // rely on automatic barrier

          Group.parallel_for_work_item([&](cl::sycl::h_item<2> It) {
            int GlobalRow = It.get_global_id()[0];
            int GlobalCol = It.get_global_id()[1];
            int Row = It.get_local_id()[0];
            int Col = It.get_local_id()[1];

            for (int K = 0; K < LSZ; K++)
              C[GlobalRow][GlobalCol] += Asub[Row][K] * Bsub[K][Col];
          });

          // rely on automatic barrier
        }
      };

      Cgh.parallel_for_work_group<class mxm_kernel>(NumGroups, BlockSize,
                                                    KernMul);
    });
    ProfInfo.push_back(Evt);
    Evt.wait();

    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<MatrixMultGroups<float>>(argc, argv);
}