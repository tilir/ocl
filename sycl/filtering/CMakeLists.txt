set(KERNELS
  filtering_sampler_nonvectorized
  filtering_sampler
  filtering_shared
  filtering_buffer
  filtering_sampler_local
  filtering_buffer_local
  filtering_sampler_local_spec
# exclude from testing 
  filtering_buffer_local_hang
)

# build kernels
foreach(KERNEL ${KERNELS})
  buildv(${KERNEL} ${KERNEL}.cc)
endforeach()

set(TESTING  
  filtering_sampler_nonvectorized
  filtering_sampler
  filtering_shared
  filtering_buffer
  filtering_sampler_local
  filtering_buffer_local
  filtering_sampler_local_spec
)

foreach(KERNEL ${TESTING})
  add_test(NAME ${KERNEL}_run
           COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL} -randboxes=512 -randfilter=3 -novis -quiet)
endforeach()