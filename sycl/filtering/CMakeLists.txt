set(KERNELS
  filtering_sampler
  filtering_sampler_vectorized
  filtering_sampler_local # really vectorized + local
  filtering_buffer
  filtering_shared_vectorized
  filtering_local # really vectorized + shared + local
)

# build kernels
foreach(KERNEL ${KERNELS})
  buildv(${KERNEL} ${KERNEL}.cc)
endforeach()

set(TESTING  
  filtering_sampler
  filtering_sampler_vectorized
  filtering_sampler_local
  filtering_buffer
  filtering_shared_vectorized
  filtering_local
)

foreach(KERNEL ${TESTING})
  add_test(NAME ${KERNEL}_run
           COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL} -randboxes=512 -randfilter=3 -novis -quiet)
endforeach()