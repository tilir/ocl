//------------------------------------------------------------------------------
//
// Rotateing with 2D convolution kernel (SYCL vs serial CPU)
// Demonstrates samplers
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

#include "rotate_testers.hpp"

using ConfigTy = sycltesters::rotate::Config;

// class is used for kernel name
class rotate_2d_sampler;

using ImAccTy = sycl::accessor<sycl::float4, 2, sycl_read, sycl_image>;
using ImWriteTy = sycl::accessor<sycl::float4, 2, sycl_write, sycl_image>;

sycl::event EnqueueRotateSampler(sycl::queue &DeviceQueue, sycl::image<2> Dst,
                                 sycl::image<2> Src, int ImW, int ImH,
                                 float Theta) {
  auto Evt = DeviceQueue.submit([&](sycl::handler &Cgh) {
    ImAccTy InPtr(Src, Cgh);
    ImWriteTy OutPtr(Dst, Cgh);
    sycl::range<2> Dims(ImW, ImH);
    sycl::sampler Sampler(sycl::coordinate_normalization_mode::unnormalized,
                          sycl::addressing_mode::clamp,
                          sycl::filtering_mode::nearest);
    float X0 = ImW / 2.0f;
    float Y0 = ImH / 2.0f;
    auto KernFilter = [=](sycl::id<2> Id) {
      const size_t Column = Id.get(0);
      const size_t Row = Id.get(1);

      float Xprime = Column - X0;
      float Yprime = Row - Y0;

      int Xr = Xprime * cos(Theta) - Yprime * sin(Theta) + X0;
      int Yr = Xprime * sin(Theta) + Yprime * cos(Theta) + Y0;

      sycl::float4 Pixel = InPtr.read(sycl::int2{Xr, Yr}, Sampler);
      OutPtr.write(sycl::int2{Column, Row}, Pixel);
    };
    Cgh.parallel_for<class rotate_2d_sampler>(Dims, KernFilter);
  });
  return Evt;
}

class RotateSamplerVec : public sycltesters::Rotate {
  using sycltesters::Rotate::Queue;
  ConfigTy Cfg_;

public:
  RotateSamplerVec(cl::sycl::queue &DeviceQueue, ConfigTy Cfg)
      : sycltesters::Rotate(DeviceQueue), Cfg_(Cfg) {}

  sycltesters::EvtRet_t operator()(sycl::float4 *DstData, sycl::float4 *SrcData,
                                   int ImW, int ImH, float Theta) override {
    sycltesters::EvtVec_t ProfInfo;
    auto &DeviceQueue = Queue();
    sycl::range<2> Dims(ImW, ImH);
    sycl::image<2> Dst(DstData, sycl_rgba, sycl_fp32, Dims);
    sycl::image<2> Src(SrcData, sycl_rgba, sycl_fp32, Dims);

    auto Evt = EnqueueRotateSampler(DeviceQueue, Dst, Src, ImW, ImH, Theta);
    ProfInfo.push_back(Evt);
    DeviceQueue.wait(); // or need host acessor to image here

    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<RotateSamplerVec>(argc, argv);
}
