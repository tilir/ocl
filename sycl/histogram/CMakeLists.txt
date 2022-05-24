set(KERNELS
  hist_naive
  hist_naive_acc
  hist_local
  hist_private
)

# build kernels
foreach(KERNEL ${KERNELS})
  buildv(${KERNEL} ${KERNEL}.cc)
endforeach()