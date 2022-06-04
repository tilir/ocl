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
  using sycltesters::Filter::Queue;
  ConfigTy Cfg_;

public:
  FilterLocalVec(cl::sycl::queue &DeviceQueue, ConfigTy Cfg)
      : sycltesters::Filter(DeviceQueue), Cfg_(Cfg) {
    if (Cfg.LocOverflow)
      throw std::runtime_error(
          "Too much local memory requested, see warning output");
  }

  sycltesters::EvtRet_t operator()(sycl::float4 *DstData, sycl::float4 *SrcData,
                                   int ImW, int ImH,
                                   drawer::Filter &Filt) override {
    sycltesters::EvtVec_t ProfInfo;
    auto &DeviceQueue = Queue();

    const int NumData = ImW * ImH;
    sycl::float4 *OutPtr = malloc_device<sycl::float4>(NumData, DeviceQueue);
    sycl::float4 *InPtr = malloc_device<sycl::float4>(NumData, DeviceQueue);
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
    // say LSZ = 4 and filter is 3x3, then we need local memory 6x6 as shown
    // below
    // ........
    // ........
    // ..xxxx..
    // ..xxxx..
    // ..xxxx..
    // ..xxxx..
    // ........
    // ........
    const int LMEM = LSZ + HalfWidth * 2;
    sycl::nd_range<2> DataSz{Dims, LDims};
    sycl::range<2> LocalMemorySize{LMEM, LMEM};
    using LTy = sycl::accessor<sycl::float4, 2, sycl_read_write, sycl_local>;

    auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
      Cgh.depends_on(EvtCpyData);
      LTy Cache{LocalMemorySize, Cgh};

      auto KernFilter = [=](sycl::nd_item<2> WorkItem) {
        const int GX = WorkItem.get_global_id(0);
        const int GY = WorkItem.get_global_id(1);
        const int LX = WorkItem.get_local_id(0);
        const int LY = WorkItem.get_local_id(1);

        // workgroup start in global
        const int WX = WorkItem.get_group(0) * LSZ;
        const int WY = WorkItem.get_group(1) * LSZ;

        // caching current pixels
        for (int I = LY; I < LMEM; I += LSZ) {
          const int Row = WY + I - HalfWidth;
          for (int J = LX; J < LMEM; J += LSZ) {
            const int Col = WX + J - HalfWidth;
            if (Row < ImH && Col < ImW && Row >= 0 && Col >= 0)
              Cache[J][I] = InPtr[Row * ImW + Col];
            else
              Cache[J][I] = 0;
          }
        }
        WorkItem.barrier(sycl_local_fence);

        // calculate convolution with cache
        sycl::float4 Sum = {0.0f, 0.0f, 0.0f, 1.0f};
        int FiltIndex = 0;

        for (int I = -HalfWidth; I <= HalfWidth; ++I) {
          for (int J = -HalfWidth; J <= HalfWidth; ++J) {
            sycl::float4 Pixel = Cache[LX + HalfWidth + J][LY + HalfWidth + I];
            Sum += Pixel * FiltPtr[FiltIndex]; // vectorized
            FiltIndex += 1;
          }
        }
        OutPtr[GY * ImW + GX] = Sum;
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
  sycltesters::test_sequence<FilterLocalVec>(argc, argv);
}
