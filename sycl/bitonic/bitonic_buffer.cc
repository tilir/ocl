//------------------------------------------------------------------------------
//
// Bitonic sort, SYCL way, with explicit buffers
// no explicit sync required
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

using ConfigTy = sycltesters::bitonicsort::Config;

template <typename T>
class BitonicSortBuf : public sycltesters::BitonicSort<T> {
  using sycltesters::BitonicSort<T>::Queue;
  ConfigTy Cfg_;

public:
  BitonicSortBuf(sycl::queue &DeviceQueue, ConfigTy Cfg)
      : sycltesters::BitonicSort<T>(DeviceQueue), Cfg_(Cfg) {}

  sycltesters::EvtRet_t operator()(T *Vec, size_t Sz) override {
    assert(Vec);
    sycltesters::EvtVec_t ProfInfo;
    if (std::popcount(Sz) != 1 || Sz < 2)
      throw std::runtime_error("Please use only power-of-two arrays");

    int N = std::countr_zero(Sz);
    sycl::buffer<T, 1> ABuf(Vec, Sz);
    auto &DeviceQueue = Queue();

    for (int Step = 0; Step < N; Step++) {
      for (int Stage = Step; Stage >= 0; Stage--) {
        sycl::range<1> NumOfItems{Sz};

        // Offload the work to kernel.
        auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
          auto A = ABuf.template get_access<sycl_read_write>(Cgh);

          auto Kernsort = [=](sycl::id<1> I) {
            const int SeqLen = 1 << (Stage + 1);
            const int Power2 = 1 << (Step - Stage);
            const int SeqNum = I / SeqLen;
            const int Odd = SeqNum / Power2;
            const bool Increasing = ((Odd % 2) == 0);
            const int HalfLen = SeqLen / 2;

            if (I < (SeqLen * SeqNum) + HalfLen) {
              const int J = I + HalfLen;
              if (((A[I] > A[J]) && Increasing) ||
                  ((A[I] < A[J]) && !Increasing)) {
                T Temp = A[I];
                A[I] = A[J];
                A[J] = Temp;
              }
            }
          };
          Cgh.parallel_for<class bitonic_sort_buf<T>>(NumOfItems, Kernsort);
        });
        ProfInfo.push_back(Evt);
        // no need for Evt.wait(), implicit dep graph will do the job
      }
    }
    DeviceQueue.wait(); // this is required to wait for the last event
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<BitonicSortBuf<int>>(argc, argv);
}
