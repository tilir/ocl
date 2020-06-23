set(HIST_SRCS
  hist.cc
)

set(NOIM_SRCS
  hist_noimage.cc
)

add_executable(hist ${HIST_SRCS})
add_executable(hist_noimage ${NOIM_SRCS})

add_dependencies(check hist_noimage)

# TODO: Intel's one not working!
# add_test(NAME "hist"
#         COMMAND ${CMAKE_CURRENT_BIN_DIR}/hist_noimage --platform Intel --program ${CMAKE_SOURCE_DIR}/kernels/hist/hist.cl --check)

add_test(NAME "hist_nv"
         COMMAND ${CMAKE_CURRENT_BIN_DIR}/hist_noimage --platform NVIDIA --program ${CMAKE_SOURCE_DIR}/kernels/hist/hist.cl --check)

# register kernels for level up
set(KERNELS ${KERNELS} hist PARENT_SCOPE)
set(KERNELS ${KERNELS} hist hist_noimage PARENT_SCOPE)
