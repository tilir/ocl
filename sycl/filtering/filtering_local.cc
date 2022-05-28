//------------------------------------------------------------------------------
//
// Filtering with 2D convolution kernel (SYCL vs serial CPU)
// Demonstrates local memory utilization
//
// This example is not perfect: it hangs on execution
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
class filter_2d_local;

class FilterLocalVec : public sycltesters::Filter {
  using sycltesters::Filter::Bundle;
  using sycltesters::Filter::Queue;
  ConfigTy Cfg_;

public:
  FilterLocalVec(cl::sycl::queue &DeviceQueue, EBundleTy ExeBundle,
                 ConfigTy Cfg)
      : sycltesters::Filter(DeviceQueue, ExeBundle), Cfg_(Cfg) {}

  sycltesters::EvtRet_t operator()(sycl::float4 *DstData, sycl::float4 *SrcData,
                                   int ImW, int ImH,
                                   drawer::Filter &Filt) override {
    sycltesters::EvtVec_t ProfInfo;
    auto &DeviceQueue = Queue();

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

    const int LSZ = Cfg_.LocSz;
    sycl::range<2> Dims(ImW, ImH);
    sycl::range<2> LDims(LSZ, LSZ);

    // we need more local memory to handle +J and -J
    const int LMEM = (LSZ + FiltSize) * (LSZ + FiltSize);
    sycl::nd_range<2> DataSz{Dims, LDims};

    auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
      Cgh.depends_on(EvtCpyData);
      using LTy = sycl::accessor<sycl::float4, 1, sycl_read_write, sycl_local>;
      sycl::range<1> LocalMemorySize{LMEM};
      LTy Cache{LocalMemorySize, Cgh};

      auto KernFilter = [=](sycl::nd_item<2> WorkItem) {
        const int GX = WorkItem.get_global_id(0);
        const int GY = WorkItem.get_global_id(1);
        const int LX = WorkItem.get_local_id(0);
        const int LY = WorkItem.get_local_id(1);

        const int RowOffset = GY * ImW;
        const int N = GX + RowOffset;

        const int LocalRowLen = HalfWidth * 2 + LSZ;
        const int LocalRowOffset = (LY + HalfWidth) * LocalRowLen;
        const int L = LocalRowOffset + LX + HalfWidth;

        // caching current pixel
        Cache[L] = InPtr[N];

        if (GX < HalfWidth || GX > ImW - HalfWidth - 1 || GY < HalfWidth ||
            GY > ImH - HalfWidth - 1) {
          // no computation for this pixel, sync and exit
          WorkItem.barrier(sycl_local_fence);
          return;
        }

        // copy additional elements
        int LocalColOffset = -1;
        int GlobalColOffset = -1;

        if (LX < HalfWidth) {
          LocalColOffset = LX;
          GlobalColOffset = -HalfWidth;
          Cache[LocalRowOffset + LX] = InPtr[N - HalfWidth];
        } else if (LX >= LSZ - HalfWidth) {
          LocalColOffset = LX + HalfWidth * 2;
          GlobalColOffset = HalfWidth;
          Cache[L + HalfWidth] = InPtr[N + HalfWidth];
        }

        if (LY < HalfWidth) {
          Cache[LY * LocalRowLen + LX + HalfWidth] = InPtr[N - HalfWidth * ImW];
          if (LocalColOffset > 0)
            Cache[LY * LocalRowLen + LocalColOffset] =
                InPtr[N - HalfWidth * ImW + GlobalColOffset];
        } else if (LY >= LSZ - HalfWidth) {
          const int Offset = (LY + HalfWidth * 2) * LocalRowLen;
          Cache[Offset + LX + HalfWidth] = InPtr[N + HalfWidth * ImW];
          if (LocalColOffset > 0)
            Cache[Offset + LocalColOffset] =
                InPtr[N + HalfWidth * ImW + GlobalColOffset];
        }

        // wait additional elements to be copied
        WorkItem.barrier(sycl_local_fence);

        // calculate convolution with cache
        sycl::float4 Sum = {0.0f, 0.0f, 0.0f, 1.0f};
        int FiltIndex = 0;
        for (int I = -HalfWidth; I <= HalfWidth; ++I) {
          for (int J = -HalfWidth; J <= HalfWidth; ++J) {
            sycl::float4 Pixel = Cache[L + I * LocalRowLen + J];
            Sum += Pixel * FiltPtr[FiltIndex]; // vectorized
            FiltIndex += 1;
          }
        }
        OutPtr[N] = Sum;
      };

      Cgh.parallel_for<class filter_2d_local>(DataSz, KernFilter);
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
  sycl::kernel_id kid = sycl::get_kernel_id<class filter_2d_local>();
  sycltesters::test_sequence<FilterLocalVec>(argc, argv, kid);
}
