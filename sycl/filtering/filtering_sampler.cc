//------------------------------------------------------------------------------
//
// Filtering with 2D convolution kernel (SYCL vs serial CPU)
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

#include "filtering_testers.hpp"

using ConfigTy = sycltesters::filter::Config;

// class is used for kernel name
class filter_2d;

class FilterSampler : public sycltesters::Filter {
  using sycltesters::Filter::Queue;
  ConfigTy Cfg_;

public:
  FilterSampler(cl::sycl::queue &DeviceQueue, EBundleTy ExeBundle, ConfigTy Cfg)
      : sycltesters::Filter(DeviceQueue, ExeBundle), Cfg_(Cfg) {}

  sycltesters::EvtRet_t operator()(sycl::image<2> Dst, sycl::image<2> Src) override {
    sycltesters::EvtVec_t ProfInfo;
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycl::kernel_id kid = sycl::get_kernel_id<filter_2d>();
  sycltesters::test_sequence<FilterSampler>(argc, argv, kid);
}
