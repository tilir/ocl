//------------------------------------------------------------------------------
//
// Filtering with 2D convolution kernel (SYCL vs serial CPU)
// Demonstrates non-samplers, just shared memory + some vectorization
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

#include "filtering_testers.hpp"

using ConfigTy = sycltesters::filter::Config;

// class is used for kernel name
class filter_2d_shared;

class FilterSharedVec : public sycltesters::Filter {
  using sycltesters::Filter::Queue;
  ConfigTy Cfg_;

public:
  FilterSharedVec(cl::sycl::queue &DeviceQueue, ConfigTy Cfg)
      : sycltesters::Filter(DeviceQueue), Cfg_(Cfg) {}

  sycltesters::EvtRet_t operator()(sycl::float4 *DstData, sycl::float4 *SrcData,
                                   int ImW, int ImH,
                                   drawer::Filter &Filt) override {
    sycltesters::EvtVec_t ProfInfo;
    auto &DeviceQueue = Queue();
    sycl::range<2> Dims(ImW, ImH);

    const int NumData = ImW * ImH;
    sycl::float4 *OutPtr = malloc_shared<sycl::float4>(NumData, DeviceQueue);
    sycl::float4 *InPtr = malloc_shared<sycl::float4>(NumData, DeviceQueue);
    auto EvtCpyData = DeviceQueue.copy(SrcData, InPtr, NumData);
    ProfInfo.emplace_back(EvtCpyData, "Copy to device");

    int FiltSize = Filt.sqrt_size();
    int DataSize = FiltSize * FiltSize;
    int HalfWidth = FiltSize / 2;

    // vectorize filter
    sycl::float4 *FiltPtr = malloc_shared<sycl::float4>(DataSize, DeviceQueue);
    const float *FiltData = Filt.data();
    for (int I = 0; I < DataSize; ++I) {
      float FiltChannel = FiltData[I];
      FiltPtr[I] = sycl::float4{FiltChannel, FiltChannel, FiltChannel, 1.0f};
    }

    auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
      Cgh.depends_on(EvtCpyData);

      auto KernFilter = [=](sycl::id<2> Id) {
        const size_t Column = Id.get(0);
        const size_t Row = Id.get(1);
        sycl::float4 Sum = {0.0f, 0.0f, 0.0f, 1.0f};
        int FiltIndex = 0;
        for (int I = -HalfWidth; I <= HalfWidth; ++I) {
          for (int J = -HalfWidth; J <= HalfWidth; ++J) {
            const auto X = Column + J;
            const auto Y = Row + I;
            // we have this in-bounds check for image processing
            // sampler does it behind the scenes (in hardware really)
            if (X >= 0 && X < ImW && Y >= 0 && Y < ImH) {
              sycl::float4 Pixel = InPtr[Y * ImW + X];
              Sum += Pixel * FiltPtr[FiltIndex]; // vectorized
            }
            FiltIndex += 1;
          }
        }
        OutPtr[Row * ImW + Column] = Sum;
      };

      Cgh.parallel_for<class filter_2d_shared>(Dims, KernFilter);
    });
    ProfInfo.emplace_back(Evt, "Convolution");

    auto EvtCpyBack = DeviceQueue.copy(OutPtr, DstData, NumData, Evt);
    ProfInfo.emplace_back(EvtCpyBack, "Copy Back");
    DeviceQueue.wait();
    sycl::free(InPtr, DeviceQueue);
    sycl::free(OutPtr, DeviceQueue);
    sycl::free(FiltPtr, DeviceQueue);
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<FilterSharedVec>(argc, argv);
}
