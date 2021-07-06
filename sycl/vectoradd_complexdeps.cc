//------------------------------------------------------------------------------
//
// Vector addition, SYCL way
// More complex dep graph with three kernels
//
//------------------------------------------------------------------------------
//
// This file is licensed after LGPL v3
// Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
//
//------------------------------------------------------------------------------

#include <CL/sycl.hpp>

#include <iostream>
#include <vector>

#include "testers.hpp"

auto host_ptr = cl::sycl::property::buffer::use_host_ptr{};

// classes used for kernel names
template <typename T> class vector_add;
template <typename T> class vector_add_scaled23;
template <typename T> class vector_add_scaled57;

template <typename T>
class VectorAddComplex : public sycltesters::VectorAdd<T> {
  using sycltesters::VectorAdd<T>::Queue;

public:
  VectorAddComplex(cl::sycl::queue &DeviceQueue)
      : sycltesters::VectorAdd<T>(DeviceQueue) {}

  sycltesters::EvtRet_t operator()(T const *AVec, T const *BVec, T *CVec,
                                   size_t Sz) override {
    std::vector<cl::sycl::event> ProfInfo;
    auto &DeviceQueue = Queue();
    cl::sycl::range<1> numOfItems{Sz};
    cl::sycl::buffer<T, 1> bufferA{AVec, numOfItems, {host_ptr}};
    cl::sycl::buffer<T, 1> bufferB{BVec, numOfItems, {host_ptr}};
    cl::sycl::buffer<T, 1> bufferC{CVec, numOfItems, {host_ptr}};
    cl::sycl::buffer<T, 1> bufferX{numOfItems};
    cl::sycl::buffer<T, 1> bufferY{numOfItems};

    bufferA.set_final_data(nullptr);
    bufferB.set_final_data(nullptr);
    bufferX.set_final_data(nullptr);
    bufferY.set_final_data(nullptr);

    auto EvtA = DeviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto A = bufferA.template get_access<sycl_read>(cgh);
      auto B = bufferB.template get_access<sycl_read>(cgh);
      auto X = bufferX.template get_access<sycl_write>(cgh);

    cgh.parallel_for<class vector_add<T>>(numOfItems, [=](cl::sycl::id<1> wiID) {
      X[wiID] = A[wiID] + B[wiID];
    });
    });
    ProfInfo.push_back(EvtA);

    auto EvtB = DeviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto A = bufferA.template get_access<sycl_read>(cgh);
      auto B = bufferB.template get_access<sycl_read>(cgh);
      auto Y = bufferY.template get_access<sycl_write>(cgh);

    cgh.parallel_for<class vector_add_scaled23<T>>(numOfItems, [=](cl::sycl::id<1> wiID) {
      Y[wiID] = 2 * A[wiID] + 3 * B[wiID];
    });
    });
    ProfInfo.push_back(EvtB);

    auto EvtC = DeviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto X = bufferX.template get_access<sycl_read>(cgh);
      auto Y = bufferY.template get_access<sycl_read>(cgh);
      auto C = bufferC.template get_access<sycl_write>(cgh);

    cgh.parallel_for<class vector_add_scaled57<T>>(numOfItems, [=](cl::sycl::id<1> wiID) {
      C[wiID] = 5 * X[wiID] + 7 * Y[wiID];
    });
    });
    ProfInfo.push_back(EvtC);

    // Host accessor (note get_access without params)
    auto A = bufferA.template get_access<sycl_read>();
    auto B = bufferB.template get_access<sycl_read>();
    auto C = bufferC.template get_access<sycl_read>();

// host-side test that one vadd iteration is correct
#ifdef VERIFY
    for (int i = 0; i < Sz; ++i)
      if (C[i] != 19 * A[i] + 26 * B[i]) {
        std::cerr << "At index: " << i << ". ";
        std::cerr << C[i] << " != " << 2 * (A[i] + B[i]) << "\n";
        abort();
      }
#endif
    return ProfInfo;
  }
};

int main(int argc, char **argv) {
  sycltesters::test_sequence<VectorAddComplex<int>>(argc, argv);
}
