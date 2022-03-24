set(KERNELS
  bitonicsort
)

# build kernels
foreach(KERNEL ${KERNELS})
  buildv(${KERNEL} ${KERNEL}.cc)
endforeach()

# special parameters
# buildv(kernel source.cc "SOME_VAR=1")

