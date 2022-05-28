set(KERNELS
  filtering_sampler
  filtering_sampler_vectorized
  filtering_buffer
  filtering_shared_vectorized
  filtering_local
)

# build kernels
foreach(KERNEL ${KERNELS})
  buildv(${KERNEL} ${KERNEL}.cc)
endforeach()