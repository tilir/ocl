if(BUILD_CIMG_KERNELS)
  find_package(PNG REQUIRED)
  include_directories(${PNG_INCLUDE_DIR})

  # this is not a kernel, so do not add to kernels list
  add_executable(triangles_demo cimg_triangles.cc)
  target_link_libraries(triangles_demo ${X11_LIBRARIES})
  target_link_libraries(triangles_demo ${PNG_LIBRARY})
  target_link_libraries(triangles_demo Threads::Threads)
endif(BUILD_CIMG_KERNELS)