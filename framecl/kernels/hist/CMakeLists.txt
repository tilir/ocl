set(HIST_SRCS
  hist.cc
)

set(NOIM_SRCS
  hist_noimage.cc
)

if(BUILD_CIMG_KERNELS)
  add_executable(hist ${HIST_SRCS})
endif(BUILD_CIMG_KERNELS)

add_executable(hist_noimage ${NOIM_SRCS})

add_dependencies(check hist_noimage)

if(INCLUDE_INTEL_TESTS)
  if(INCLUDE_FAILING_TESTS)
    add_test(NAME "hist"
             COMMAND ${CMAKE_CURRENT_BIN_DIR}/hist_noimage --platform Intel --program ${CMAKE_SOURCE_DIR}/kernels/hist/hist.cl --check)
  endif(INCLUDE_FAILING_TESTS)
endif(INCLUDE_INTEL_TESTS)

if(INCLUDE_NVIDIA_TESTS)
  add_test(NAME "hist_nv"
           COMMAND ${CMAKE_CURRENT_BIN_DIR}/hist_noimage --platform NVIDIA --program ${CMAKE_SOURCE_DIR}/kernels/hist/hist.cl --check)
endif(INCLUDE_NVIDIA_TESTS)

# register kernels for level up
set(ADDKERNELS hist_noimage)
if(BUILD_CIMG_KERNELS)
  set(ADDKERNELS ${ADDKERNELS} hist)
endif(BUILD_CIMG_KERNELS)

set(KERNELS ${KERNELS} ${ADDKERNELS} PARENT_SCOPE)