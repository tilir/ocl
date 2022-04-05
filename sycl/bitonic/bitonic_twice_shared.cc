//------------------------------------------------------------------------------
//
// Bitonic sort, SYCL way, with shared memory
// Experiment of offloading even more work: both internal loops
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
  BitonicSortShared(cl::sycl::queue &DeviceQueue, unsigned Lsz)
      : sycltesters::BitonicSort<T>(DeviceQueue), Lsz_(Lsz) {}

  sycltesters::EvtRet_t operator()(T *Vec, size_t Sz) override {
    assert(Vec);
    std::vector<cl::sycl::event> ProfInfo;
    if (std::popcount(Sz) != 1 || Sz < 2)
      throw std::runtime_error("Please use only power-of-two arrays");

    int N = std::countr_zero(Sz);
    auto &DeviceQueue = Queue();

#ifdef HOST_ALLOC
    T *A = cl::sycl::malloc_host<T>(Sz, DeviceQueue);
#else
    T *A = cl::sycl::malloc_shared<T>(Sz, DeviceQueue);
#endif
    std::copy(A, A + Sz, Vec);

    cl::sycl::range<1> NumOfItems{Sz};
    cl::sycl::range<1> BlockSize{Lsz_};
    cl::sycl::nd_range<1> Range{NumOfItems, BlockSize};

    auto Evt = DeviceQueue.submit([&](cl::sycl::handler &Cgh) {
      auto Kernsort = [=](cl::sycl::nd_item<1> Item) {
        int I = Item.get_global_id(0);
        for (int Step = 0; Step < N; Step++) {
          for (int Stage = Step; Stage >= 0; Stage--) {
            int SeqLen = 1 << (Stage + 1);
            int Power2 = 1 << (Step - Stage);
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
          }
        }
      };
      Cgh.parallel_for<class bitonic_sort_shared<T>>(Range, Kernsort);
    });
    ProfInfo.push_back(Evt);
    Evt.wait();
    std::copy(Vec, Vec + Sz, A);
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<BitonicSortShared<int>>(argc, argv);
}
