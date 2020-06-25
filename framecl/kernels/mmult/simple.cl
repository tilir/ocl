__kernel void matrix_multiply(
    __global int *A, __global int *B, __global int *C, int AX, int AY, int BY) {
  int row = get_global_id(0);
  int col = get_global_id(1);
  int sum = 0;

  for (int k = 0; k < AY; k++)
    sum += A[row * AY + k] * B[k * BY + col];

  C[row * BY + col] = sum;
}