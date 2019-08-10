#include <cassert>
#include <iostream>

#include "cl_wrapper2.h"

constexpr unsigned long BIG_SZ = 256 * 16;

const char *rkernel = STRINGIFY(__kernel void reduce(
    __global int *input, __global int *output, __local int *reductionSums) {
  const int globalID = get_global_id(0);
  const int localID = get_local_id(0);
  const int localSize = get_local_size(0);
  const int workgroupID = get_group_id(0);

  reductionSums[localID] = input[globalID];

  for (int offset = localSize / 2; offset > 0; offset /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (localID < offset)
      reductionSums[localID] += reductionSums[localID + offset];
  }

  if (localID == 0)
    output[workgroupID] = reductionSums[0];
});

int main() {
  std::vector<int> a(BIG_SZ);
  for (int i = 0; i < BIG_SZ; ++i)
    a[i] = i;

  oclwrap2::ocl_app_t app;
  std::cout << "Selected platform: " << app.platform_version() << std::endl;
  std::cout << "Selected device: " << app.device_name() << std::endl;

  int pidx = app.add_programm(rkernel);
  int kidx = app.extract_kernel(pidx, "reduce");

  size_t bsz = BIG_SZ;
  size_t wgsz = app.max_workgroup_size();
  std::cout << "Selected work group size: " << wgsz << std::endl;

  auto nres = bsz / wgsz;
  std::vector<int> results(nres);

  int ibuf = app.add_buffer<int>(CL_MEM_READ_ONLY, a.data(), BIG_SZ);
  int obuf = app.add_buffer<int>(CL_MEM_WRITE_ONLY, NULL, nres);

  app.set_kernel_buf_arg(kidx, 0, ibuf);
  app.set_kernel_buf_arg(kidx, 1, obuf);
  app.set_kernel_localbuf_arg(kidx, 2, wgsz * sizeof(int));

  app.exec_kernel_nd(kidx, 1, &bsz, &wgsz);
  app.read_buffer<int>(obuf, results.data(), nres);

  long sum = 0;
  for (int i = 0; i < nres; ++i) {
    std::cout << "results[" << i << "] = " << results[i] << std::endl;
    sum += results[i];
  }

  std::cout << "result: " << sum << std::endl;
  std::cout << "shall be: " << BIG_SZ * (BIG_SZ - 1) / 2 << std::endl;
}
