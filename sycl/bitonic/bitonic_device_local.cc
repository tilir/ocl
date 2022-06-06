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
template <typename T> class bitonic_device_local_steps;
template <typename T> class bitonic_device_local_stages;
template <typename T> class bitonic_device_global;

using ConfigTy = sycltesters::bitonicsort::Config;

template <typename T>
auto EnqueueLocalIterationLastStages(sycl::queue &DeviceQueue, T *A, int GSZ,
                                     int LSZ, int Step, int StageStart) {
  sycl::range<1> GRange{GSZ}, LRange{LSZ};
  sycl::nd_range<1> IterSpace(GRange, LRange);
  using LTy = sycl::accessor<T, 1, sycl_read_write, sycl_local>;
  const int LMEM = LSZ;
  sycl::range<1> LocalMemorySize{LMEM};

  auto Evt = DeviceQueue.submit([=](sycl::handler &Cgh) {
    LTy Cache{LocalMemorySize, Cgh};
    sycl::stream Out(1024, 256, Cgh);

    auto KernStages = [=](sycl::nd_item<1> WorkItem) {
      const int G = WorkItem.get_global_id(0);
      const int L = WorkItem.get_local_id(0);
      Cache[L] = A[G];
      WorkItem.barrier();

      const int I = L;
      int Stage = StageStart;
      for (int Stage = StageStart; Stage >= 0; Stage--) {
        const int SeqLen = 1 << (Stage + 1);
        const int Power2 = 1 << (Step - Stage);
        const int SeqNum = I / SeqLen;
        // direction determined by global position, not local
        const int Odd = (G / SeqLen) / Power2;
        const bool Increasing = ((Odd % 2) == 0);
        const int HalfLen = SeqLen / 2;

        if (I < SeqLen * SeqNum + HalfLen) {
          const int J = I + HalfLen;
          if (((Cache[I] > Cache[J]) && Increasing) ||
              ((Cache[I] < Cache[J]) && !Increasing)) {
            const T Temp = Cache[I];
            Cache[I] = Cache[J];
            Cache[J] = Temp;
          }
        }
        WorkItem.barrier();
      }
      A[G] = Cache[L];
    };
    Cgh.parallel_for<class bitonic_device_local_stages<T>>(IterSpace,
                                                           KernStages);
  });
  return Evt;
}

template <typename T>
auto EnqueueLocalIterationFirstSteps(sycl::queue &DeviceQueue, T *A, int GSZ,
                                     int LSZ, int StepStart, int StepEnd) {
  sycl::range<1> GRange{GSZ}, LRange{LSZ};
  sycl::nd_range<1> IterSpace(GRange, LRange);
  using LTy = sycl::accessor<T, 1, sycl_read_write, sycl_local>;
  const int LMEM = LSZ;
  sycl::range<1> LocalMemorySize{LMEM};
  auto Evt = DeviceQueue.submit([=](sycl::handler &Cgh) {
    LTy Cache{LocalMemorySize, Cgh};
    auto KernSteps = [=](sycl::nd_item<1> WorkItem) {
      const int G = WorkItem.get_global_id(0);
      const int L = WorkItem.get_local_id(0);
      Cache[L] = A[G];
      WorkItem.barrier(sycl_local_fence);

      const int I = L;
      for (int Step = StepStart; Step < StepEnd; Step++) {
        for (int Stage = Step; Stage >= 0; Stage--) {
          const int SeqLen = 1 << (Stage + 1);
          const int Power2 = 1 << (Step - Stage);
          const int SeqNum = I / SeqLen;
          // direction determined by global position, not local
          const int Odd = (G / SeqLen) / Power2;
          const bool Increasing = ((Odd % 2) == 0);
          const int HalfLen = SeqLen / 2;

          if (I < SeqLen * SeqNum + HalfLen) {
            const int J = I + HalfLen;
            if (((Cache[I] > Cache[J]) && Increasing) ||
                ((Cache[I] < Cache[J]) && !Increasing)) {
              const T Temp = Cache[I];
              Cache[I] = Cache[J];
              Cache[J] = Temp;
            }
          }
          WorkItem.barrier(sycl_local_fence);
        }
      }
      A[G] = Cache[L];
    };
    Cgh.parallel_for<class bitonic_device_local_steps<T>>(IterSpace, KernSteps);
  });
  return Evt;
}

template <typename T>
class BitonicDeviceLocal : public sycltesters::BitonicSort<T> {
  using sycltesters::BitonicSort<T>::Queue;
  ConfigTy Cfg_;

public:
  BitonicDeviceLocal(sycl::queue &DeviceQueue, ConfigTy Cfg)
      : sycltesters::BitonicSort<T>(DeviceQueue), Cfg_(Cfg) {}

  sycltesters::EvtRet_t operator()(T *Vec, size_t Sz) override {
    assert(Vec);
    const unsigned LSZ = Cfg_.LocSz;
    const unsigned GSZ = Sz;
    sycltesters::EvtVec_t ProfInfo;
    if (std::popcount(Sz) != 1 || Sz < 2)
      throw std::runtime_error("Please use only power-of-two arrays");
    if (std::popcount(LSZ) != 1 || LSZ < 2)
      throw std::runtime_error("Please use only power-of-two local sizes");

    int N = std::countr_zero(Sz);
    int NFST = std::countr_zero(LSZ);
    auto &DeviceQueue = Queue();

    // set silent inside op()
    auto OldState = sycltesters::qout.state();
    if (!Cfg_.Verbose)
      sycltesters::qout.set(true);

    T *A = sycl::malloc_device<T>(Sz, DeviceQueue);
    auto EvtCpyData = DeviceQueue.copy(Vec, A, Sz);
    ProfInfo.emplace_back(EvtCpyData, "Copy to device");
    EvtCpyData.wait();

    sycltesters::qout << "N = " << N << std::endl;
    sycltesters::qout << "NFST = " << NFST << std::endl;
    sycltesters::qout << "GSZ = " << GSZ << std::endl;
    sycltesters::qout << "LSZ = " << LSZ << std::endl;

    auto Evt = EnqueueLocalIterationFirstSteps(DeviceQueue, A, Sz, LSZ, 0,
                                               std::min(NFST - 1, N));
    ProfInfo.emplace_back(Evt, "Starting iterations");
    Evt.wait(); // no implicit task graph

    // visualization after prep
    if (Cfg_.Verbose) {
      sycltesters::qout << "After local pre-sort: " << NFST - 1 << std::endl;
      DeviceQueue.copy(A, Vec, Sz);
      DeviceQueue.wait();
      visualize_seq(Vec, Vec + Sz, sycltesters::qout);
    }

    sycl::range<1> NumOfItems{Sz};
    for (int Step = NFST - 1; Step < N; Step++) {
      int StageLast = NFST - 2;

      for (int Stage = Step; Stage >= StageLast; Stage--) {
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

          Cgh.parallel_for<class bitonic_device_global<T>>(NumOfItems,
                                                           Kernsort);
        });
        ProfInfo.emplace_back(Evt, "Next iteration");
        Evt.wait(); // no implicit task graph

        // visualization after stage
        if (Cfg_.Verbose) {
          sycltesters::qout << "After stage: " << Stage << std::endl;
          DeviceQueue.copy(A, Vec, Sz);
          DeviceQueue.wait();
          visualize_seq(Vec, Vec + Sz, sycltesters::qout);
        }
      }

      // schedule all stages up to (Step - NFST) as small ones
      auto Evt = EnqueueLocalIterationLastStages(DeviceQueue, A, Sz, LSZ, Step,
                                                 StageLast - 1);
      ProfInfo.emplace_back(Evt, "Starting stages for next step");
      Evt.wait(); // no implicit task graph

      if (Cfg_.Verbose) {
        sycltesters::qout << "After local stages: " << StageLast << std::endl;
        DeviceQueue.copy(A, Vec, Sz);
        DeviceQueue.wait();
        visualize_seq(Vec, Vec + Sz, sycltesters::qout);
      }
    }

    // only for Sz <= LMEM
    auto EvtCpyBack = DeviceQueue.copy(A, Vec, Sz);
    ProfInfo.emplace_back(EvtCpyBack, "Copy back");
    DeviceQueue.wait();
    sycl::free(A, DeviceQueue);

    // restoring if it was changed
    sycltesters::qout.set(OldState);
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<BitonicDeviceLocal<int>>(argc, argv);
}
