set(KERNELS
  bitonicsort
  bitonic_shared
)

# build kernels
foreach(KERNEL ${KERNELS})
  buildv(${KERNEL} ${KERNEL}.cc)
endforeach()

# special parameters
buildv(bitonic_simple bitonicsort.cc "SIMPLERANGE=1")
buildv(bitonic_host bitonic_shared.cc "HOST_ALLOC=1")