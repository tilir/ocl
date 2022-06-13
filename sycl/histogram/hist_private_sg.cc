//------------------------------------------------------------------------------
//
// Histogram with smaller private memory blocks (SYCL vs serial CPU).
// In this example SYCL uses less private memory because we are using subgroups
//
// But it is principally buggy, and not clear how to fix general case
// For instance:
// * works fine as: -bsz=1 -sz=2048 -gsz=1024 -hsz=64 -lsz=16
// * fails at: -bsz=1 -sz=2048 -gsz=2048 -hsz=64 -lsz=16
// etc...
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

#include "hist_testers.hpp"

using ConfigTy = sycltesters::hist::Config;

// class is used for kernel name
template <typename T> class hist_subgroup_private_shared;

template <typename T>
class HistogrammSubgroupPrivateShared : public sycltesters::Histogramm<T> {
  using sycltesters::Histogramm<T>::Queue;
  ConfigTy Cfg_;

public:
  HistogrammSubgroupPrivateShared(sycl::queue &DeviceQueue, ConfigTy Cfg)
      : sycltesters::Histogramm<T>(DeviceQueue), Cfg_(Cfg) {}

  sycltesters::EvtRet_t operator()(const T *Data, T *Bins, int NumData,
                                   int NumBins) override {
    assert(Data != nullptr && Bins != nullptr);
    constexpr int MAX_HSZ = 256;
    if ((MAX_HSZ < NumBins) || ((Cfg_.GlobSz % NumBins) != 0)) {
      std::cerr << "Now this example works only if #Bins <= " << MAX_HSZ
                << std::endl;
      std::cerr << "Also #Bins shall divide global size: #Bins = " << NumBins
                << ", GSZ = " << Cfg_.GlobSz << std::endl;
      std::terminate();
    }
    const auto NGSZ = Cfg_.GlobSz / NumBins;
    const int LSZ = Cfg_.LocSz;
    sycltesters::EvtVec_t ProfInfo;
    auto &DeviceQueue = Queue();
    auto *BufferData = sycl::malloc_shared<T>(NumData, DeviceQueue);
    auto *BufferBins = sycl::malloc_shared<T>(NumBins, DeviceQueue);
    std::copy(Data, Data + NumData, BufferData);
    std::fill(BufferBins, BufferBins + NumBins, 0);
    sycl::range<1> DataSz{NGSZ}, LocSz{LSZ};
    sycl::nd_range IterSpace(DataSz, LocSz);

    constexpr int SGSize = 16;

    auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
      sycl::stream Out(4096, 256, Cgh);
      auto KernHist = [=](sycl::nd_item<1> Id)
          [[sycl::reqd_sub_group_size(SGSize)]] {
        T PrivateHist[MAX_HSZ / SGSize] = {0};
        const int N = Id.get_global_id(0);
        const int G = Id.get_group(0);
        const auto SubGroup = Id.get_sub_group();
        const int SGI = SubGroup.get_group_id()[0];
        const int SLI = SubGroup.get_local_id()[0];
        const int SLR = SubGroup.get_local_range()[0];
        // currently expected GR == SGSize otherwise PrivateHist size will fail
        // problem is: this may be not the case...

        // this is really useful snippet for in-depth understanding
        // it will output: BufferData[0], BufferData[64], BufferData[1],
        // BufferData[65], etc..
        //
        // if (N < 3) {
        //   const T VData = SubGroup.load(BufferData + G * LSZ + SGI * SLR);
        //   Out << N << ": I = " << N << ", VData = " << VData << "\n";
        //   const T VData64 = SubGroup.load(BufferData + G * LSZ + SGI * SLR +
        //   NumBins); Out << N << ": I = " << N + NumBins << ", VData = " <<
        //   VData64 << "\n";
        // }

        // building private histograms
        for (int I = N; I < NumData / NumBins; I += NGSZ) {
          for (int J = 0; J < NumBins; J += 1) {
            // read current element
            const int Base = G * LSZ + SGI * SLR;
            const auto VData =
                SubGroup.load(BufferData + I * NumBins + Base + J * SGSize);
#pragma unroll
            for (int K = 0; K < SGSize; K++) {
              const auto Y = sycl::group_broadcast(SubGroup, VData, K);
              // Y / SGSIZE is bin number
              if (SLI == (Y % SGSize))
                PrivateHist[Y / SGSize] += 1;
            }
          }
        }

        // combining all private histograms
        for (int I = 0; I < NumBins / SGSize; I += 1) {
          const T Data = PrivateHist[I];
          global_atomic_ref<T>(BufferBins[SGSize * I + SLI]).fetch_add(Data);
        }
      };

      Cgh.parallel_for<class hist_subgroup_private_shared<T>>(IterSpace,
                                                              KernHist);
    });

    ProfInfo.emplace_back(Evt, "Calculate histogramm");

    // copy back (note dependency on Evt)
    auto EvtCpyBins = DeviceQueue.copy(BufferBins, Bins, NumBins, Evt);
    ProfInfo.emplace_back(EvtCpyBins, "Copy bins back");
    DeviceQueue.wait();

    sycl::free(BufferData, DeviceQueue);
    sycl::free(BufferBins, DeviceQueue);
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<HistogrammSubgroupPrivateShared<int>>(argc, argv);
}
