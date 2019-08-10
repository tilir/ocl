//-----------------------------------------------------------------------------
//
// Simple OpenCL 2.x wrapper class
//
// in ctor we are picking first 2.x platform and best device to create context
// then every string kernel accepted
// in dtor everything is released
//
//-----------------------------------------------------------------------------
//
// This file is licensed after GNU GPL v3
//
//-----------------------------------------------------------------------------

#pragma once

#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <CL/cl.h>

#define STRINGIFY(...) #__VA_ARGS__

#define CHECK_ERR(ret) cl_process_error(ret, __FILE__, __LINE__)

namespace oclwrap2 {

void cl_process_error(cl_int ret, const char *file, int line);

class ocl_app_t {  
  int nplatform_;

  cl_platform_id platform_id_;
  cl_device_id device_id_;
  cl_context context_;
  cl_command_queue command_queue_;

  std::vector<cl_platform_id> platforms_;
  std::vector<std::vector<cl_device_id>> devices_;

  std::vector<std::pair<cl_mem, size_t>> memobjs_;
  std::vector<cl_program> programs_;
  std::vector<cl_kernel> kernels_;
  std::vector<cl_event> events_;
  std::vector<cl_sampler> samplers_;

public:
  ocl_app_t();
  ~ocl_app_t();
  int add_programm(std::string codeline);
  int extract_kernel(int pidx, const char *kname);
  void release_mems();

  // add and (if buf != NULL) initialize buffer
  template <typename T>
  int add_buffer(cl_mem_flags flags, const T *buf, size_t size);

  // add buffer and fill it with value
  template <typename T> int add_buffer(cl_mem_flags flags, size_t size, T init);

  // create image and (maybe) fill it with value
  template <typename T>
  int add_2d_image(int w, int h, int s, T *buf, cl_channel_order ord,
                   cl_channel_type ctp, cl_mem_flags flags);

  int add_sampler(cl_bool normalized_coords, cl_addressing_mode amode, cl_filter_mode filter);
  int add_pipe(cl_mem_flags flags, cl_uint psize, cl_uint pcount);

  void set_kernel_buf_arg(int kidx, int narg, int bidx);
  void set_kernel_int_arg(int kidx, int narg, int arg);
  void set_kernel_float_arg(int kidx, int narg, float arg);
  void set_kernel_localbuf_arg(int kidx, int narg, int argsz);
  void set_kernel_sampler_arg(int kidx, int narg, int sidx);

  // inorder interface (without events)
public:
  template <typename T>
  void write_buffer(int bidx, const T *src, size_t size) const;

  template <typename T> void read_buffer(int bidx, T *dst, size_t size) const;

  template <typename T>
  void read_2d_image(int imidx, T *dst, size_t x, size_t y) const;

  void exec_kernel_nd(int kidx, int nd, size_t *globalws,
                      size_t *localws) const;

  // public selectors
public:
  std::string platform_version() const;
  std::string device_name() const;
  size_t max_workgroup_size() const;
  void dump_devices(std::ostream &os) const;

  // construction helpers
private:
  void init_devices();
  std::string get_platform_param_str(cl_platform_id pid,
                                     cl_platform_info pname) const;
  int select_platform() const;

  template <typename T>
  T get_device_param_scalar(cl_device_id devid, cl_device_info pname) const;

  std::string get_device_param_str(cl_device_id devid,
                                   cl_device_info pname) const;
  cl_device_id select_device() const;
};

std::string read_programm(const char *fname);

template <typename T>
int ocl_app_t::add_buffer(cl_mem_flags flags, const T *buf, size_t size) {
  cl_int ret;
  auto bufsz = size * sizeof(T);
  cl_mem mem_obj = clCreateBuffer(context_, flags, bufsz, NULL, &ret);
  CHECK_ERR(ret);
  memobjs_.push_back(std::make_pair(mem_obj, bufsz));
  size_t bufidx = memobjs_.size() - 1;
  if (buf != NULL) {
    ret = clEnqueueWriteBuffer(command_queue_, mem_obj, CL_TRUE, 0, bufsz, buf,
                               0, NULL, NULL);
    CHECK_ERR(ret);
  }
  return bufidx;
}

template <typename T>
int ocl_app_t::add_buffer(cl_mem_flags flags, size_t size, T init) {
  cl_int ret;
  auto bufsz = size * sizeof(T);
  cl_mem mem_obj = clCreateBuffer(context_, flags, bufsz, NULL, &ret);
  CHECK_ERR(ret);
  memobjs_.push_back(std::make_pair(mem_obj, bufsz));
  size_t bufidx = memobjs_.size() - 1;
  cl_event completion;
  ret = clEnqueueFillBuffer(command_queue_, mem_obj, &init, sizeof(init), 0,
                            bufsz, 0, NULL, &completion);
  CHECK_ERR(ret);
  ret = clWaitForEvents(1, &completion);
  CHECK_ERR(ret);
  return bufidx;
}

template <typename T>
int ocl_app_t::add_2d_image(int w, int h, int s, T *buf, cl_channel_order ord,
                            cl_channel_type ctp, cl_mem_flags flags) {
  cl_int ret;
  cl_image_desc desc;
  memset(&desc, 0, sizeof(cl_image_desc));
  desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  desc.image_width = w;
  desc.image_height = h;
  desc.buffer = NULL;

  cl_image_format format;
  format.image_channel_order = ord;
  format.image_channel_data_type = ctp;

  cl_mem img = clCreateImage(context_, flags, &format, &desc, NULL, &ret);
  CHECK_ERR(ret);

  if (buf != NULL) {
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {static_cast<size_t>(w), static_cast<size_t>(h), 1};
    ret = clEnqueueWriteImage(command_queue_, img, CL_TRUE, origin, region, 0,
                              0, buf, 0, NULL, NULL);
    CHECK_ERR(ret);
  }

  memobjs_.push_back(std::make_pair(img, w * h * s * sizeof(T)));
  return memobjs_.size() - 1;
}

template <typename T>
void ocl_app_t::write_buffer(int bidx, const T *src, size_t size) const {
  cl_int ret;
  auto bufsz = size * sizeof(T);
  auto mo = memobjs_.at(bidx);
  assert(mo.second == size);
  ret = clEnqueueWriteBuffer(command_queue_, mo.first, CL_TRUE, 0, bufsz, src,
                             0, NULL, NULL);
  CHECK_ERR(ret);
}

template <typename T>
void ocl_app_t::read_buffer(int bidx, T *dst, size_t size) const {
  cl_int ret;
  auto bufsz = size * sizeof(T);
  auto mo = memobjs_.at(bidx);
  assert(mo.second == bufsz);
  ret = clEnqueueReadBuffer(command_queue_, mo.first, CL_TRUE, 0, bufsz, dst, 0,
                            NULL, NULL);
  CHECK_ERR(ret);
}

template <typename T>
void ocl_app_t::read_2d_image(int imidx, T *dst, size_t x, size_t y) const {
  cl_int ret;
  auto mo = memobjs_.at(imidx);
  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {x, y, 1};
  ret = clEnqueueReadImage(command_queue_, mo.first, CL_TRUE, origin, region, 0,
                           0, dst, 0, NULL, NULL);
  CHECK_ERR(ret);
}

template <typename T>
T ocl_app_t::get_device_param_scalar(cl_device_id devid,
                                     cl_device_info pname) const {
  cl_int ret;
  T result;
  ret = clGetDeviceInfo(devid, pname, sizeof(result), &result, NULL);
  CHECK_ERR(ret);
  return result;
}

} // namespace oclwrap2

