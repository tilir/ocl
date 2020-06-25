__kernel void reduce(
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
}
