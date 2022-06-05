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

using ConfigTy = sycltesters::bitonicsort::Config;

template <typename T>
class BitonicSortShared : public sycltesters::BitonicSort<T> {
  using sycltesters::BitonicSort<T>::Queue;
  ConfigTy Cfg_;

public:
  BitonicSortShared(sycl::queue &DeviceQueue, ConfigTy Cfg)
      : sycltesters::BitonicSort<T>(DeviceQueue), Cfg_(Cfg) {}

  sycltesters::EvtRet_t operator()(T *Vec, size_t Sz) override {
    assert(Vec);
    sycltesters::EvtVec_t ProfInfo;
    if (std::popcount(Sz) != 1 || Sz < 2)
      throw std::runtime_error("Please use only power-of-two arrays");

    int N = std::countr_zero(Sz);
    auto &DeviceQueue = Queue();

    T *A = sycl::malloc_device<T>(Sz, DeviceQueue);
    auto EvtCpyData = DeviceQueue.copy(Vec, A, Sz);
    ProfInfo.emplace_back(EvtCpyData, "Copy to device");
    EvtCpyData.wait();
    for (int Step = 0; Step < N; Step++) {
      for (int Stage = Step; Stage >= 0; Stage--) {
        sycl::range<1> NumOfItems{Sz};

        // Offload the work to kernel.
        auto Evt = DeviceQueue.submit([=](sycl::handler &Cgh) {
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

          Cgh.parallel_for<class bitonic_sort_shared<T>>(NumOfItems, Kernsort);
        });
        ProfInfo.emplace_back(Evt, "Next iteration");
        Evt.wait(); // no implicit task graph
      }
    }
    auto EvtCpyBack = DeviceQueue.copy(A, Vec, Sz);
    ProfInfo.emplace_back(EvtCpyBack, "Copy back");
    DeviceQueue.wait();
    sycl::free(A, DeviceQueue);
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<BitonicSortShared<int>>(argc, argv);
}
