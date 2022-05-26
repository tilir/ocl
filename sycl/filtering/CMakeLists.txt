set(KERNELS
  filtering_sampler
)

# build kernels
foreach(KERNEL ${KERNELS})
  buildv(${KERNEL} ${KERNEL}.cc)
endforeach()