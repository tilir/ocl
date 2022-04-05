set(KERNELS
  bitonicsort
  bitonic_shared
  bitonic_loop_shared
)

# build kernels
foreach(KERNEL ${KERNELS})
  buildv(${KERNEL} ${KERNEL}.cc)
endforeach()

# special parameters
buildv(bitonic_host bitonic_shared.cc "HOST_ALLOC=1")
buildv(bitonic_loop_host bitonic_loop_shared.cc "HOST_ALLOC=1")

