//-----------------------------------------------------------------------------
//
// Simple OpenCL 2.x wrapper class
//
//-----------------------------------------------------------------------------
//
// This file is licensed after GNU GPL v3
//
//-----------------------------------------------------------------------------

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>

#include "cl_wrapper2.h"

namespace oclwrap2 {

static void cl_notify_fn(const char *errinfo, const void *private_info,
                         size_t cb, void *user_data);

//-----------------------------------------------------------------------------
//
// ocl_app_t public interface (creation/destruction)
//
//-----------------------------------------------------------------------------

ocl_app_t::ocl_app_t() {
  cl_int ret;
  platform_id_ = select_platform();
  device_id_ = select_device();

  context_ = clCreateContext(NULL, 1, &device_id_, cl_notify_fn, NULL, &ret);
  CHECK_ERR(ret);

#if (CL_TARGET_OPENCL_VERSION > 120)
  command_queue_ =
      clCreateCommandQueueWithProperties(context_, device_id_, NULL, &ret);
#else
  command_queue_ = clCreateCommandQueue(context_, device_id_, 0, &ret);
#endif
  CHECK_ERR(ret);
}

void ocl_app_t::release_mems() {
  cl_int ret;
  for (auto obj : memobjs_) {
    ret = clReleaseMemObject(obj.first);
    CHECK_ERR(ret);
  }
  memobjs_.clear();
}

ocl_app_t::~ocl_app_t() {
  cl_int ret;
  ret = clFlush(command_queue_);
  CHECK_ERR(ret);
  ret = clFinish(command_queue_);
  CHECK_ERR(ret);

  for (auto krn : kernels_) {
    ret = clReleaseKernel(krn);
    CHECK_ERR(ret);
  }

  for (auto prog : programs_) {
    ret = clReleaseProgram(prog);
    CHECK_ERR(ret);
  }

  release_mems();

  ret = clReleaseCommandQueue(command_queue_);
  CHECK_ERR(ret);
  ret = clReleaseContext(context_);
  CHECK_ERR(ret);
}

int ocl_app_t::add_programm(std::string codeline) {
  cl_int ret;
  const char *cstr = codeline.c_str();
  const size_t csz = codeline.size();
  cl_program p = clCreateProgramWithSource(context_, 1, &cstr, &csz, &ret);
  CHECK_ERR(ret);
  ret = clBuildProgram(p, 1, &device_id_, NULL, NULL, NULL);
  if (ret != CL_SUCCESS) {
    cl_int oldret = ret;
    size_t sz;
    ret = clGetProgramBuildInfo(p, device_id_, CL_PROGRAM_BUILD_LOG, 0, NULL,
                                &sz);
    CHECK_ERR(ret);
    std::cerr << "Build log size = " << sz << std::endl;
    std::vector<char> buf(sz);
    ret = clGetProgramBuildInfo(p, device_id_, CL_PROGRAM_BUILD_LOG, sz,
                                buf.data(), NULL);
    CHECK_ERR(ret);
    std::string logname = "build.log";
    {
      std::ofstream of(logname, std::ofstream::binary);
      of.write(buf.data(), sz);
    }
    std::cerr << "Program build log is in: " << logname << std::endl;
    CHECK_ERR(oldret);
  }
  programs_.push_back(p);
  return programs_.size() - 1;
}

int ocl_app_t::extract_kernel(int pidx, const char *kname) {
  cl_int ret;
  cl_program p = programs_.at(pidx);
  cl_kernel k = clCreateKernel(p, kname, &ret);
  CHECK_ERR(ret);
  kernels_.push_back(k);
  return kernels_.size() - 1;
}

int ocl_app_t::add_sampler(cl_bool normalized_coords, cl_addressing_mode amode,
                           cl_filter_mode filter) {
  cl_int ret;
  cl_sampler s;

#if (CL_TARGET_OPENCL_VERSION > 120)
  cl_sampler_properties properties[] = {CL_SAMPLER_NORMALIZED_COORDS,
                                        normalized_coords,
                                        CL_SAMPLER_ADDRESSING_MODE,
                                        amode,
                                        CL_SAMPLER_FILTER_MODE,
                                        filter,
                                        0};
  s = clCreateSamplerWithProperties(context_, properties, &ret);
#else
  s = clCreateSampler(context_, normalized_coords, amode, filter, &ret);
#endif

  CHECK_ERR(ret);
  samplers_.push_back(s);
  return samplers_.size() - 1;
}

void ocl_app_t::set_kernel_buf_arg(int kidx, int narg, int bidx) {
  cl_int ret;
  cl_kernel k = kernels_.at(kidx);
  cl_mem mo = memobjs_.at(bidx).first;
  ret = clSetKernelArg(k, narg, sizeof(cl_mem), static_cast<void *>(&mo));
  CHECK_ERR(ret);
}

void ocl_app_t::set_kernel_int_arg(int kidx, int narg, int arg) {
  cl_int ret;
  cl_kernel k = kernels_.at(kidx);
  ret = clSetKernelArg(k, narg, sizeof(int), static_cast<void *>(&arg));
  CHECK_ERR(ret);
}

void ocl_app_t::set_kernel_float_arg(int kidx, int narg, float arg) {
  cl_int ret;
  cl_kernel k = kernels_.at(kidx);
  ret = clSetKernelArg(k, narg, sizeof(float), static_cast<void *>(&arg));
  CHECK_ERR(ret);
}

void ocl_app_t::set_kernel_localbuf_arg(int kidx, int narg, int argsz) {
  cl_int ret;
  cl_kernel k = kernels_.at(kidx);
  ret = clSetKernelArg(k, narg, argsz, NULL);
  CHECK_ERR(ret);
}

void ocl_app_t::set_kernel_sampler_arg(int kidx, int narg, int sidx) {
  cl_int ret;
  cl_kernel k = kernels_.at(kidx);
  cl_sampler s = samplers_.at(sidx);
  ret = clSetKernelArg(k, narg, sizeof(cl_sampler), static_cast<void *>(&s));
  CHECK_ERR(ret);
}

//-----------------------------------------------------------------------------
//
// ocl_app_t public interface (operations)
//
//-----------------------------------------------------------------------------

void ocl_app_t::exec_kernel_nd(int kidx, int nd, size_t *globalsz,
                               size_t *localsz) const {
  cl_int ret;
  cl_kernel k = kernels_.at(kidx);
  cl_event completion;
  ret = clEnqueueNDRangeKernel(command_queue_, k, nd, NULL, globalsz, localsz,
                               0, NULL, &completion);
  CHECK_ERR(ret);
  ret = clWaitForEvents(1, &completion);
  CHECK_ERR(ret);
}

//-----------------------------------------------------------------------------
//
// ocl_app_t public selectors
//
//-----------------------------------------------------------------------------

std::string ocl_app_t::platform_version() const {
  return get_platform_param_str(platform_id_, CL_PLATFORM_VERSION);
}

std::string ocl_app_t::device_name() const {
  return get_device_param_str(device_id_, CL_DEVICE_NAME);
}

size_t ocl_app_t::max_workgroup_size() const {
  cl_int ret;
  size_t res;
  ret = clGetDeviceInfo(device_id_, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                        sizeof(size_t), &res, NULL);
  CHECK_ERR(ret);
  return res;
}

//-----------------------------------------------------------------------------
//
// ocl_app_t construction helpers
//
//-----------------------------------------------------------------------------

std::string ocl_app_t::get_platform_param_str(cl_platform_id pid,
                                              cl_platform_info pname) const {
  size_t memsize;
  std::string result;
  cl_int ret;
  ret = clGetPlatformInfo(pid, pname, 0, NULL, &memsize);
  CHECK_ERR(ret);
  result.resize(memsize);
  ret = clGetPlatformInfo(pid, pname, memsize,
                          const_cast<char *>(result.data()), NULL);
  CHECK_ERR(ret);
  return result;
}

cl_platform_id ocl_app_t::select_platform() const {
  cl_uint nplatforms;
  std::vector<cl_platform_id> ids;

  cl_int ret = clGetPlatformIDs(0, NULL, &nplatforms);
  CHECK_ERR(ret);
  if (nplatforms < 1)
    throw std::runtime_error("There shall be at least one platform");

  ids.resize(nplatforms);
  ret = clGetPlatformIDs(nplatforms, ids.data(), NULL);
  CHECK_ERR(ret);

  auto supported_it =
      std::find_if(ids.begin(), ids.end(), [this](cl_platform_id pid) {
        std::string ver = get_platform_param_str(pid, CL_PLATFORM_VERSION);
        std::stringstream ss;
        ss << ver;
        std::string ocl;
        int major;
        ss >> ocl >> major;
#if (CL_TARGET_OPENCL_VERSION > 120)
        return (major >= 2);
#else
          return (major < 2);
#endif
      });

  if (supported_it == ids.end())
    throw std::runtime_error("No supported OpenCL 2.0 platform");

  return *supported_it;
}

std::string ocl_app_t::get_device_param_str(cl_device_id devid,
                                            cl_device_info pname) const {
  cl_int ret;
  size_t memsize;
  std::string result;
  ret = clGetDeviceInfo(devid, pname, 0, NULL, &memsize);
  CHECK_ERR(ret);
  result.resize(memsize);
  ret = clGetDeviceInfo(devid, pname, memsize,
                        const_cast<char *>(result.data()), NULL);
  CHECK_ERR(ret);
  return result;
}

cl_device_id ocl_app_t::select_device() const {
  cl_int ret;
  cl_uint numdevices;
  ret = clGetDeviceIDs(platform_id_, CL_DEVICE_TYPE_ALL, 0, NULL, &numdevices);
  CHECK_ERR(ret);
  if (numdevices < 1)
    throw std::runtime_error("There shall be at least one device");
  std::vector<cl_device_id> possible_devs(numdevices);
  ret = clGetDeviceIDs(platform_id_, CL_DEVICE_TYPE_ALL, numdevices,
                       possible_devs.data(), NULL);
  CHECK_ERR(ret);

  // selecting one with max #of CUs or max size of WG
  auto seldev_it =
      std::max_element(possible_devs.begin(), possible_devs.end(),
                       [this](cl_device_id fst, cl_device_id snd) {
#if defined(MAXCU_STRATEGY)
                         return get_device_param_scalar<cl_uint>(
                                    fst, CL_DEVICE_MAX_COMPUTE_UNITS) <
                                get_device_param_scalar<cl_uint>(
                                    snd, CL_DEVICE_MAX_COMPUTE_UNITS);
#else
                         return get_device_param_scalar<size_t>(
                                    fst, CL_DEVICE_MAX_WORK_GROUP_SIZE) <
                                get_device_param_scalar<size_t>(
                                    snd, CL_DEVICE_MAX_WORK_GROUP_SIZE);
#endif
                       });

  if (seldev_it == possible_devs.end())
    throw std::runtime_error("No device chosen");

  return *seldev_it;
}

//-----------------------------------------------------------------------------
//
// Free functions
//
//-----------------------------------------------------------------------------

void cl_process_error(cl_int ret, const char *file, int line) {
  const char *cause = "unknown";

  if (ret == CL_SUCCESS) {
    return;
  }
#undef PROCESS
#define PROCESS(ret, STR)                                                      \
  else if (ret == STR) {                                                       \
    cause = #STR;                                                              \
  }
  PROCESS(ret, CL_DEVICE_NOT_FOUND)
  PROCESS(ret, CL_DEVICE_NOT_AVAILABLE)
  PROCESS(ret, CL_COMPILER_NOT_AVAILABLE)
  PROCESS(ret, CL_MEM_OBJECT_ALLOCATION_FAILURE)
  PROCESS(ret, CL_OUT_OF_RESOURCES)
  PROCESS(ret, CL_OUT_OF_HOST_MEMORY)
  PROCESS(ret, CL_OUT_OF_HOST_MEMORY)
  PROCESS(ret, CL_PROFILING_INFO_NOT_AVAILABLE)
  PROCESS(ret, CL_MEM_COPY_OVERLAP)
  PROCESS(ret, CL_IMAGE_FORMAT_MISMATCH)
  PROCESS(ret, CL_IMAGE_FORMAT_NOT_SUPPORTED)
  PROCESS(ret, CL_BUILD_PROGRAM_FAILURE)
  PROCESS(ret, CL_MAP_FAILURE)
  PROCESS(ret, CL_MISALIGNED_SUB_BUFFER_OFFSET)
  PROCESS(ret, CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
  PROCESS(ret, CL_COMPILE_PROGRAM_FAILURE)
  PROCESS(ret, CL_LINKER_NOT_AVAILABLE)
  PROCESS(ret, CL_LINK_PROGRAM_FAILURE)
  PROCESS(ret, CL_DEVICE_PARTITION_FAILED)
  PROCESS(ret, CL_KERNEL_ARG_INFO_NOT_AVAILABLE)
  PROCESS(ret, CL_INVALID_VALUE)
  PROCESS(ret, CL_INVALID_DEVICE_TYPE)
  PROCESS(ret, CL_INVALID_PLATFORM)
  PROCESS(ret, CL_INVALID_DEVICE)
  PROCESS(ret, CL_INVALID_VALUE)
  PROCESS(ret, CL_INVALID_DEVICE_TYPE)
  PROCESS(ret, CL_INVALID_PLATFORM)
  PROCESS(ret, CL_INVALID_DEVICE)
  PROCESS(ret, CL_INVALID_CONTEXT)
  PROCESS(ret, CL_INVALID_QUEUE_PROPERTIES)
  PROCESS(ret, CL_INVALID_COMMAND_QUEUE)
  PROCESS(ret, CL_INVALID_HOST_PTR)
  PROCESS(ret, CL_INVALID_MEM_OBJECT)
  PROCESS(ret, CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
  PROCESS(ret, CL_INVALID_IMAGE_SIZE)
  PROCESS(ret, CL_INVALID_SAMPLER)
  PROCESS(ret, CL_INVALID_BINARY)
  PROCESS(ret, CL_INVALID_BUILD_OPTIONS)
  PROCESS(ret, CL_INVALID_PROGRAM)
  PROCESS(ret, CL_INVALID_PROGRAM_EXECUTABLE)
  PROCESS(ret, CL_INVALID_KERNEL_NAME)
  PROCESS(ret, CL_INVALID_KERNEL_DEFINITION)
  PROCESS(ret, CL_INVALID_KERNEL)
  PROCESS(ret, CL_INVALID_ARG_INDEX)
  PROCESS(ret, CL_INVALID_ARG_VALUE)
  PROCESS(ret, CL_INVALID_ARG_SIZE)
  PROCESS(ret, CL_INVALID_KERNEL_ARGS)
  PROCESS(ret, CL_INVALID_WORK_DIMENSION)
  PROCESS(ret, CL_INVALID_WORK_GROUP_SIZE)
  PROCESS(ret, CL_INVALID_WORK_ITEM_SIZE)
  PROCESS(ret, CL_INVALID_GLOBAL_OFFSET)
  PROCESS(ret, CL_INVALID_EVENT_WAIT_LIST)
  PROCESS(ret, CL_INVALID_EVENT)
  PROCESS(ret, CL_INVALID_OPERATION)
  PROCESS(ret, CL_INVALID_GL_OBJECT)
  PROCESS(ret, CL_INVALID_BUFFER_SIZE)
  PROCESS(ret, CL_INVALID_MIP_LEVEL)
  PROCESS(ret, CL_INVALID_GLOBAL_WORK_SIZE)
  PROCESS(ret, CL_INVALID_PROPERTY)
  PROCESS(ret, CL_INVALID_IMAGE_DESCRIPTOR)
  PROCESS(ret, CL_INVALID_COMPILER_OPTIONS)
  PROCESS(ret, CL_INVALID_LINKER_OPTIONS)
  PROCESS(ret, CL_INVALID_DEVICE_PARTITION_COUNT)
#if (CL_TARGET_OPENCL_VERSION > 120)
  PROCESS(ret, CL_INVALID_PIPE_SIZE)
  PROCESS(ret, CL_INVALID_DEVICE_QUEUE)
#endif
#undef PROCESS
  else {
    cause = "Unknown";
  }

  std::stringstream ss;
  ss << "Error " << cause << " at " << file << ":" << line << " code " << ret
     << std::endl;
  throw std::runtime_error(ss.str().c_str());
}

static void cl_notify_fn(const char *errinfo, const void *private_info,
                         size_t cb, void *user_data) {
  std::stringstream ss;
  std::ofstream pfile("context.err", std::ofstream::binary);
  pfile.write(static_cast<const char *>(private_info), cb);
  ss << "Context error " << errinfo << std::endl;
  throw std::runtime_error(ss.str().c_str());
}

std::string read_programm(const char *fname) {
  std::stringstream ptext;
  std::ifstream pfile(fname);

  for (;;) {
    std::string s;
    std::getline(pfile, s);
    if (!pfile)
      break;
    ptext << s << std::endl;
  }

  return ptext.str();
}

} // namespace oclwrap2
