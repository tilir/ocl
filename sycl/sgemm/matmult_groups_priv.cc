//------------------------------------------------------------------------------
//
// Matrix multiplication with explicit parallel for workitem
// Also using private memory here to speed up things
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
template <typename T> class mmult_groups_priv;

using ConfigTy = sycltesters::sgemm::Config;

template <typename T>
class MatrixMultGroups : public sycltesters::MatrixMult<T> {
  using sycltesters::MatrixMult<T>::Queue;
  ConfigTy Cfg_;

public:
  MatrixMultGroups(sycl::queue &DeviceQueue, ConfigTy Cfg)
      : sycltesters::MatrixMult<T>(DeviceQueue), Cfg_(Cfg) {}

  sycltesters::EvtRet_t operator()(const T *Aptr, const T *Bptr, T *Cptr,
                                   size_t AX, size_t AY, size_t BY) override {
    const auto LSZ = Cfg_.Lsz; // avoid implicit capture of this
    assert(Aptr != nullptr && Bptr != nullptr && Cptr != nullptr);
    if ((AY % LSZ) != 0)
      throw std::runtime_error("Expect local size = multiple of AY");
    sycltesters::EvtVec_t ProfInfo;
    sycl::range<2> Asz{AX, AY}, Bsz{AY, BY}, Csz{AX, BY};
    sycl::buffer<T, 2> BufA(Aptr, Asz), BufB(Bptr, Bsz), BufC(Cptr, Csz);

    BufA.set_final_data(nullptr);
    BufB.set_final_data(nullptr);

    auto &DeviceQueue = Queue();
    auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
      auto A = BufA.template get_access<sycl_read>(Cgh);
      auto B = BufB.template get_access<sycl_read>(Cgh);
      auto C = BufC.template get_access<sycl_write>(Cgh);

      sycl::range<2> NumGroups{AX / LSZ, BY / LSZ};
      sycl::range<2> BlockSize{LSZ, LSZ};
      const int NumTiles = AY / LSZ;

      // local memory
      using LTy = sycl::accessor<T, 2, sycl_read_write, sycl_local>;
      LTy Asub{BlockSize, Cgh}, Bsub{BlockSize, Cgh};

      auto KernMul = [=](sycl::group<2> Group) {
        sycl::private_memory<int, 2> Sum(Group);

        Group.parallel_for_work_item([&](sycl::h_item<2> It) { Sum(It) = 0; });

        for (int Tile = 0; Tile < NumTiles; Tile++) {
          Group.parallel_for_work_item([&](sycl::h_item<2> It) {
            int GlobalRow = It.get_global_id(0);
            int GlobalCol = It.get_global_id(1);
            int Row = It.get_local_id(0);
            int Col = It.get_local_id(1);
            int TiledRow = LSZ * Tile + Row;
            int TiledCol = LSZ * Tile + Col;

            Asub[Row][Col] = A[GlobalRow][TiledCol];
            Bsub[Row][Col] = B[TiledRow][GlobalCol];
          });

          // rely on automatic barrier

          Group.parallel_for_work_item([&](sycl::h_item<2> It) {
            int Row = It.get_local_id(0);
            int Col = It.get_local_id(1);

            for (int K = 0; K < LSZ; K++)
              Sum(It) += Asub[Row][K] * Bsub[K][Col];
          });

          // rely on automatic barrier
        }

        // now assign sum to its cell
        Group.parallel_for_work_item([&](sycl::h_item<2> It) {
          int GlobalRow = It.get_global_id(0);
          int GlobalCol = It.get_global_id(1);
          C[GlobalRow][GlobalCol] = Sum(It);
        });
      };

      Cgh.parallel_for_work_group<class mmult_groups_priv<T>>(
          NumGroups, BlockSize, KernMul);
    });
    ProfInfo.push_back(Evt);
    Evt.wait();

    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<MatrixMultGroups<float>>(argc, argv);
}