//------------------------------------------------------------------------------
//
// Basically SYCL have 4 programming models (SYCL 1.2.1, 3.6)
//   * Data parallel kernels
//   * Work-group data parallel kernels
//   * Hierarchical data parallel kernels
//   * Kernels that are not launched over parallel instances
//
// Also there are 4 types of index space classes
//   * cl::sycl::id      -- contains a single (global) id
//   * cl::sycl::item    -- contains the global id and local id
//   * cl::sycl::nd_item -- contains the global id, local id and work-group id
//   * cl::sycl::group   -- contains work-group id and methods on wg
//
// This file shows everything
// It is intentionally straightforward
//
// This implementation uses streams
//
// try -DWGSERR to invalid work group size
// try -DUSEGPU to use GPU and thus unsync stream
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

#ifndef SIZE
#define SIZE 1024 * 10
#endif

constexpr auto sycl_atomic = cl::sycl::access::mode::atomic;
constexpr auto sycl_read = cl::sycl::access::mode::read;
constexpr auto sycl_read_write = cl::sycl::access::mode::read_write;

class simple_dpc_1d;
class wg_dpc_1d;
class hier_dpc_1d;

int main() {
  // with GPU selector, stream does not need to be synchronized
#ifdef USEGPU
  cl::sycl::gpu_selector gpsel;
  cl::sycl::queue deviceQueue{gpsel};
#else
  cl::sycl::host_selector hsel;
  cl::sycl::queue deviceQueue{hsel};
#endif

  std::cout << "Welcome to SYCL programming models" << std::endl;
  std::cout << "Selected device:" << std::endl;

  // some device information
  auto device = deviceQueue.get_device();
  std::cout << "  " << device.get_info<cl::sycl::info::device::name>()
            << std::endl;
  std::cout << "  "
            << "Driver version: "
            << device.get_info<cl::sycl::info::device::driver_version>()
            << std::endl;
  std::cout << "  "
            << device.get_info<cl::sycl::info::device::opencl_c_version>()
            << std::endl;
  std::cout << "Compute configuration for " << SIZE
            << " elements:" << std::endl;

  size_t work_group_size =
      device.get_info<cl::sycl::info::device::max_work_group_size>();
  size_t num_work_groups = (SIZE + work_group_size - 1) / work_group_size;
  std::cout << "  work group size:  " << work_group_size << std::endl;
  std::cout << "  number of groups: " << num_work_groups << std::endl;

  {
    std::cout << "--- offload #1 ---" << std::endl;

    // simples possible buffer for just 1 element
    int counter = 0;
    auto counter_buf = cl::sycl::buffer<int, 1>(&counter, 1);

    // offload #1
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      // atomic access to counter_buf
      auto counter_d = counter_buf.get_access<sycl_atomic>(cgh);
      cl::sycl::stream out(4096, 256, cgh);

      // data parallel kernel with single 1d range parameter
      cgh.parallel_for<class simple_dpc_1d>(
          cl::sycl::range<1>{SIZE}, [=](cl::sycl::id<1> work_item) {
            int wid = work_item[0];            

            // we can not use stdout, because accessing non-const global
            // variable is not allowed within SYCL device code
            if ((wid % 500) == 0)
              out << "id: " << wid << cl::sycl::endl;
            cl::sycl::atomic_fetch_add(counter_d[0], 1);
          });
    });

    auto counter_h = counter_buf.get_access<sycl_read>();
    std::cout << "counter: " << counter_h[0] << std::endl;
    std::cout << "expected counter: " << SIZE << std::endl;
  }

  {
    std::cout << "--- offload #2 ---" << std::endl;

    // simplest possible buffer for just 1 element
    int counter = 0;
    auto counter_buf = cl::sycl::buffer<int, 1>(&counter, 1);

    // offload #2
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      // atomic access to counter_buf
      auto counter_d = counter_buf.get_access<sycl_atomic>(cgh);
      cl::sycl::stream out(4096, 256, cgh);

      // workgroup data parallel kernel with 1d nd_range parameter and
      // global/local ranges
      //
      // btw try here work_group_size - 1 for silent error
      auto wgs = work_group_size;
      #ifdef WGSERR
        wgs -= 1;
      #endif
      cgh.parallel_for<class wg_dpc_1d>(
          cl::sycl::nd_range<1>{cl::sycl::range<1>(SIZE),
                                cl::sycl::range<1>(wgs)},
          [=](cl::sycl::nd_item<1> work_item) {
            int gid = work_item.get_global_id(0);
            int lid = work_item.get_local_id(0);
            int wid = work_item.get_group(0);
            if ((gid % 500) == 0)
              out << "gid: " << gid << ", lid: " << lid << ", wid: " << wid << cl::sycl::endl;
            cl::sycl::atomic_fetch_add(counter_d[0], 1);
          });
    });

    auto counter_h = counter_buf.get_access<sycl_read>();
    std::cout << "counter: " << counter_h[0] << std::endl;
    std::cout << "expected counter: " << SIZE << std::endl;
  }

  {
    std::cout << "--- offload #3 ---" << std::endl;

    // simplest possible buffer for just 1 element
    int counter = 0;
    auto counter_buf = cl::sycl::buffer<int, 1>(&counter, 1);

    // offload #3
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      // atomic access to counter_buf
      auto counter_d = counter_buf.get_access<sycl_atomic>(cgh);
      cl::sycl::stream out(4096, 256, cgh);

      // hierarchical data parallel kernel with 1d nd_range parameter and
      // explicit group/item loops
      cgh.parallel_for_work_group<class hier_dpc_1d>(
          cl::sycl::range<1>{num_work_groups},
          cl::sycl::range<1>{work_group_size}, [=](cl::sycl::group<1> group) {
            int wid = group.get_id(0);
            group.parallel_for_work_item([&](cl::sycl::h_item<1> item) {
              int gid = item.get_global_id(0);
              if ((gid % 500) == 0)
                out << "group id: " << wid << " gid: " << gid << cl::sycl::endl;
              cl::sycl::atomic_fetch_add(counter_d[0], 1);
            });
          });
    });

    auto counter_h = counter_buf.get_access<sycl_read>();
    std::cout << "counter: " << counter_h[0] << std::endl;
    std::cout << "expected counter: " << SIZE << std::endl;
  }
}
