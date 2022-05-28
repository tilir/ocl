//------------------------------------------------------------------------------
//
// Bitonic sort, SYCL way, with shared memory
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

#include "bitonic_testers.hpp"

// class is used for kernel name
template <typename T> class bitonic_sort_shared;

template <typename T>
class BitonicSortShared : public sycltesters::BitonicSort<T> {
  using sycltesters::BitonicSort<T>::Queue;
  unsigned Lsz_;

public:
  BitonicSortShared(sycl::queue &DeviceQueue, unsigned Lsz)
      : sycltesters::BitonicSort<T>(DeviceQueue), Lsz_(Lsz) {}

  sycltesters::EvtRet_t operator()(T *Vec, size_t Sz) override {
    assert(Vec);
    sycltesters::EvtVec_t ProfInfo;
    if (std::popcount(Sz) != 1 || Sz < 2)
      throw std::runtime_error("Please use only power-of-two arrays");

    int N = std::countr_zero(Sz);
    auto &DeviceQueue = Queue();

#ifdef HOST_ALLOC
    T *A = sycl::malloc_host<T>(Sz, DeviceQueue);
#else
    T *A = sycl::malloc_shared<T>(Sz, DeviceQueue);
#endif
    auto EvtCpyData = DeviceQueue.copy(Vec, A, Sz);
    EvtCpyData.wait();
    ProfInfo.emplace_back(EvtCpyData, "Copy to device");
    for (int Step = 0; Step < N; Step++) {
      for (int Stage = Step; Stage >= 0; Stage--) {
        int SeqLen = 1 << (Stage + 1);
        int Power2 = 1 << (Step - Stage);
        sycl::range<1> NumOfItems{Sz};
        sycl::range<1> BlockSize{Lsz_};
        sycl::nd_range<1> Range{NumOfItems, BlockSize};

        // Offload the work to kernel.
        auto Evt = DeviceQueue.submit([=](sycl::handler &Cgh) {
          auto Kernsort = [=](sycl::nd_item<1> Item) {
            int I = Item.get_global_id(0);
            int SeqNum = I / SeqLen;
            int Odd = SeqNum / Power2;
            bool Increasing = ((Odd % 2) == 0);
            int HalfLen = SeqLen / 2;

            if (I < (SeqLen * SeqNum) + HalfLen) {
              int J = I + HalfLen;
              if (((A[I] > A[J]) && Increasing) ||
                  ((A[I] < A[J]) && !Increasing)) {
                T Temp = A[I];
                A[I] = A[J];
                A[J] = Temp;
              }
            }
          };

          Cgh.parallel_for<class bitonic_sort_shared<T>>(Range, Kernsort);
        });
        ProfInfo.emplace_back(Evt, "Next iteration");
        Evt.wait();
      }
    }
    EvtCpyData = DeviceQueue.copy(A, Vec, Sz);
    ProfInfo.emplace_back(EvtCpyData, "Copy back");
    DeviceQueue.wait();
    sycl::free(A, DeviceQueue);
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<BitonicSortShared<int>>(argc, argv);
}
