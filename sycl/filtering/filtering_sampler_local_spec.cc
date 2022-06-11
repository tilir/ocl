//------------------------------------------------------------------------------
//
// Filtering with 2D convolution kernel (SYCL vs serial CPU)
// Demonstrates local memory utilization and specialization constants
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

// class is used for kernel name
class filter_2d_local_spec;

constexpr static sycl::specialization_id<int> HalfWidthC;
constexpr static sycl::specialization_id<int> LMEMC;
constexpr static sycl::specialization_id<int> LSZC;
constexpr static sycl::specialization_id<int> ImWC;
constexpr static sycl::specialization_id<int> ImHC;

using ConfigTy = sycltesters::filter::Config;

class FilterSamplerLocalSpec : public sycltesters::Filter {
  using sycltesters::Filter::Queue;
  ConfigTy Cfg_;

public:
  FilterSamplerLocalSpec(cl::sycl::queue &DeviceQueue, ConfigTy Cfg)
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

    const int LSZ = Cfg_.LocSz;
    const int ImWR = simplemath::roundup(ImW, LSZ);
    const int ImHR = simplemath::roundup(ImH, LSZ);

    sycl::range<2> LDims(LSZ, LSZ);
    sycl::range<2> GDims(ImWR, ImHR);
    sycl::nd_range<2> IterSpace{GDims, LDims};

    sycl::range<2> Dims(ImW, ImH);
    sycl::image<2> Dst(DstData, sycl_rgba, sycl_fp32, Dims);
    sycl::image<2> Src(SrcData, sycl_rgba, sycl_fp32, Dims);

    int FiltSize = Filt.sqrt_size();
    int DataSize = FiltSize * FiltSize;
    int HalfWidth = FiltSize / 2;

    std::vector<sycl::float4> FiltVectorized;
    const float *FiltData = Filt.data();
    for (int I = 0; I < DataSize; ++I) {
      const float FData = FiltData[I];
      FiltVectorized.emplace_back(FData, FData, FData, 1.0f);
    }
    sycl::buffer<sycl::float4, 1> FiltBuffer(FiltVectorized.data(), DataSize);

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
    sycl::range<2> LocalMemorySize{LMEM, LMEM};
    using ImAccTy = sycl::accessor<sycl::float4, 2, sycl_read, sycl_image>;
    using ImWriteTy = sycl::accessor<sycl::float4, 2, sycl_write, sycl_image>;
    using LTy = sycl::accessor<sycl::float4, 2, sycl_read_write, sycl_local>;

    sycl::kernel_id KId = sycl::get_kernel_id<filter_2d_local_spec>();
    sycl::kernel_bundle KbSrc =
        sycl::get_kernel_bundle<sycl::bundle_state::input>(
            DeviceQueue.get_context(), {KId});
    KbSrc.template set_specialization_constant<HalfWidthC>(HalfWidth);
    KbSrc.template set_specialization_constant<LMEMC>(LMEM);
    KbSrc.template set_specialization_constant<LSZC>(LSZ);
    KbSrc.template set_specialization_constant<ImWC>(ImW);
    KbSrc.template set_specialization_constant<ImHC>(ImH);
    sycl::kernel_bundle Kb = sycl::build(KbSrc);

    auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
      Cgh.use_kernel_bundle(Kb);
      LTy Cache{LocalMemorySize, Cgh};
      ImAccTy InPtr(Src, Cgh);    // image to read from
      ImWriteTy OutPtr(Dst, Cgh); // image to write to
      auto FiltPtr = FiltBuffer.template get_access<sycl_read>(Cgh);

      // sampler for image read
      sycl::sampler Sampler(sycl::coordinate_normalization_mode::unnormalized,
                            sycl::addressing_mode::clamp,
                            sycl::filtering_mode::nearest);

      auto KernFilter = [=](sycl::nd_item<2> WorkItem,
                            sycl::kernel_handler Kh) {
        const int GX = WorkItem.get_global_id(0);
        const int GY = WorkItem.get_global_id(1);
        const int LX = WorkItem.get_local_id(0);
        const int LY = WorkItem.get_local_id(1);

        const int HalfWidthK =
            Kh.template get_specialization_constant<HalfWidthC>();
        const int LMEMK = Kh.template get_specialization_constant<LMEMC>();
        const int LSZK = Kh.template get_specialization_constant<LSZC>();
        const int ImWK = Kh.template get_specialization_constant<ImWC>();
        const int ImHK = Kh.template get_specialization_constant<ImHC>();

        // workgroup start in global
        const int WX = WorkItem.get_group(0) * LSZK;
        const int WY = WorkItem.get_group(1) * LSZK;

#if 0
        // when doing this check here, everything hangs
        if (GX >= ImWK || GY >= ImHK)
          return;
#endif

// caching current pixels
#pragma unroll
        for (int I = LY; I < LMEMK; I += LSZK) {
          const int Row = WY + I - HalfWidthK;
#pragma unroll
          for (int J = LX; J < LMEMK; J += LSZK) {
            const int Col = WX + J - HalfWidthK;
            sycl::int2 Coords = {Col, Row};
            Cache[J][I] = InPtr.read(Coords, Sampler);
          }
        }
        WorkItem.barrier(sycl_local_fence);

        // we are out of image bounds
        if (GX >= ImWK || GY >= ImHK)
          return;

        // calculate convolution with cache
        sycl::float4 Sum = {0.0f, 0.0f, 0.0f, 1.0f};
        int FiltIndex = 0;

#pragma unroll
        for (int I = -HalfWidthK; I <= HalfWidthK; ++I) {
#pragma unroll
          for (int J = -HalfWidthK; J <= HalfWidthK; ++J) {
            sycl::float4 Pixel =
                Cache[LX + HalfWidthK + J][LY + HalfWidthK + I];
            Sum += Pixel * FiltPtr[FiltIndex]; // vectorized
            FiltIndex += 1;
          }
        }
        sycl::int2 Coords = {GX, GY};
        if (GX < ImWK && GY < ImHK)
          OutPtr.write(Coords, Sum);
      };

      Cgh.parallel_for<class filter_2d_local_spec>(IterSpace, KernFilter);
    });
    ProfInfo.emplace_back(Evt, "Convolution");
    DeviceQueue.wait();
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<FilterSamplerLocalSpec>(argc, argv);
}
