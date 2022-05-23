//------------------------------------------------------------------------------
//
// Bitonic sort, SYCL way, with explicit buffers
// no explicit sync required
//
// Macros to control things:
// -DSIMPLERANGE : use simple range instead of ND-range
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
template <typename T> class bitonic_sort_buf;

template <typename T>
class BitonicSortBuf : public sycltesters::BitonicSort<T> {
  using sycltesters::BitonicSort<T>::Queue;
  unsigned Lsz_;

public:
  BitonicSortBuf(cl::sycl::queue &DeviceQueue, unsigned Lsz)
      : sycltesters::BitonicSort<T>(DeviceQueue), Lsz_(Lsz) {}

  sycltesters::EvtRet_t operator()(T *Vec, size_t Sz) override {
    assert(Vec);
    sycltesters::EvtVec_t ProfInfo;
    if (std::popcount(Sz) != 1 || Sz < 2)
      throw std::runtime_error("Please use only power-of-two arrays");

    int N = std::countr_zero(Sz);
    cl::sycl::buffer<T, 1> ABuf(Vec, Sz);
    auto &DeviceQueue = Queue();

    for (int Step = 0; Step < N; Step++) {
      for (int Stage = Step; Stage >= 0; Stage--) {
        int SeqLen = 1 << (Stage + 1);
        int Power2 = 1 << (Step - Stage);
        cl::sycl::range<1> NumOfItems{Sz};
#if !defined(SIMPLERANGE)
        cl::sycl::range<1> BlockSize{Lsz_};
        cl::sycl::nd_range<1> Range{NumOfItems, BlockSize};
#endif

        // Offload the work to kernel.
        auto Evt = DeviceQueue.submit([&](cl::sycl::handler &Cgh) {
          auto AVec = ABuf.template get_access<sycl_read_write>(Cgh);

#if !defined(SIMPLERANGE)
          auto Kernsort = [=](cl::sycl::nd_item<1> Item) {
            int I = Item.get_global_id(0);
#else
          auto Kernsort = [=](cl::sycl::id<1> I) {
#endif
            int SeqNum = I / SeqLen;
            int Odd = SeqNum / Power2;
            bool Increasing = ((Odd % 2) == 0);
            int HalfLen = SeqLen / 2;

            if (I < (SeqLen * SeqNum) + HalfLen) {
              int J = I + HalfLen;
              if (((AVec[I] > AVec[J]) && Increasing) ||
                  ((AVec[I] < AVec[J]) && !Increasing)) {
                T Temp = AVec[I];
                AVec[I] = AVec[J];
                AVec[J] = Temp;
              }
            }
          };

#if !defined(SIMPLERANGE)
          Cgh.parallel_for<class bitonic_sort_buf<T>>(Range, Kernsort);
#else
          Cgh.parallel_for<class bitonic_sort_buf<T>>(NumOfItems, Kernsort);
#endif
        });
        ProfInfo.push_back(Evt);
        // no need for Evt.wait(), implicit dep graph will do the job
      }
    }
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<BitonicSortBuf<int>>(argc, argv);
}
