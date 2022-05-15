set(KERNELS
  matmult
  matmult_local
  matmult_local_shared
  matmult_groups
  matmult_groups_priv
)

# common includes and libs
foreach(KERNEL ${KERNELS})
  buildv(${KERNEL} ${KERNEL}.cc)
endforeach()

# special targets with defines
buildv(matmult_nopriv matmult.cc "NOPRIVATE=1")
buildv(matmult_local_nobarrier matmult_local.cc "NOBARRIER=1")

if(USE_MKL)
buildv(matmult_mkl matmult.cc)
buildv(matmult_mkl_trans matmult.cc "MKLTRANS=1")
target_link_libraries(matmult_mkl $<LINK_ONLY:MKL::MKL_DPCPP>)
target_link_libraries(matmult_mkl_trans $<LINK_ONLY:MKL::MKL_DPCPP>)
endif()