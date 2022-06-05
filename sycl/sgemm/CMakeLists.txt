set(KERNELS
  matmult
  matmult_specialization
  matmult_specialization_svm
  matmult_transposed
  matmult_local
  matmult_local_shared
  matmult_local_shared_spec
  matmult_groups
  matmult_groups_priv
)

# common includes and libs
foreach(KERNEL ${KERNELS})
  buildv(${KERNEL} ${KERNEL}.cc)
endforeach()

# special targets with defines
buildv(matmult_nopriv matmult.cc "NOPRIVATE=1")
buildv(matmult_notrans matmult_transposed.cc "NONTRANSPOSED=1")
# excluded from testing
buildv(matmult_local_nobarrier matmult_local.cc "NOBARRIER=1")

if(USE_MKL)
buildv(matmult_mkl matmult.cc)
buildv(matmult_mkl_trans matmult.cc "MKLTRANS=1")
target_link_libraries(matmult_mkl $<LINK_ONLY:MKL::MKL_DPCPP>)
target_link_libraries(matmult_mkl_trans $<LINK_ONLY:MKL::MKL_DPCPP>)
endif()

set(TESTING  
  matmult
  matmult_transposed
  matmult_notrans
  matmult_specialization
  matmult_specialization_svm
  matmult_local
  matmult_local_shared
  matmult_local_shared_spec
  matmult_groups
  matmult_groups_priv
  matmult_nopriv
)

if(USE_MKL)
list(APPEND TESTING matmult_mkl)
list(APPEND TESTING matmult_mkl_trans)
endif()

foreach(KERNEL ${TESTING})
  add_test(NAME ${KERNEL}_run
           COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL} -quiet)
endforeach()