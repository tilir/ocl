set(KERNELS
  filtering_sampler
  filtering_sampler_vectorized
)

# build kernels
foreach(KERNEL ${KERNELS})
  buildv(${KERNEL} ${KERNEL}.cc)
endforeach()