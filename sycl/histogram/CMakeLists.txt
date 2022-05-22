set(KERNELS
  hist_naive
  hist_naive_acc
  hist_local
)

# build kernels
foreach(KERNEL ${KERNELS})
  buildv(${KERNEL} ${KERNEL}.cc)
endforeach()