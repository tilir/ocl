set(KERNELS
  vectoradd
  vectoradd_complexdeps
  vectoradd_devicemem
  vectoradd_sharedmem
  vectoradd_wait
  vectoradd_inequal
  vectoradd_simd
)

# build kernels
foreach(KERNEL ${KERNELS})
  buildv(${KERNEL} ${KERNEL}.cc)
endforeach()

# special parameters
buildv(vectoradd_hostmem vectoradd_sharedmem.cc "HOST_ALLOC=1")
buildv(vectoradd_inord vectoradd_wait.cc "INORD=1")
buildv(vectoradd_nowait vectoradd_wait.cc "NOWAIT=1")
buildv(vectoradd_inord_nowait vectoradd_wait.cc "NOWAIT=1" "INORD=1")
