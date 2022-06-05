//------------------------------------------------------------------------------
//
// Useful short aliases for SYCL stuff
//
//------------------------------------------------------------------------------
//
// This file is licensed after LGPL v3
// Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
//
//------------------------------------------------------------------------------

#pragma once

#include <CL/sycl.hpp>

// access modes
constexpr auto sycl_read = sycl::access::mode::read;
constexpr auto sycl_write = sycl::access::mode::write;
constexpr auto sycl_read_write = sycl::access::mode::read_write;
constexpr auto sycl_atomic = sycl::access::mode::atomic;

// targets for accessors
constexpr auto sycl_global = sycl::access::target::device;
constexpr auto sycl_local = sycl::access::target::local;
constexpr auto sycl_constant = cl::sycl::access::target::constant_buffer;
constexpr auto sycl_image = sycl::access::target::image;

// fences
constexpr auto sycl_local_fence = sycl::access::fence_space::local_space;
constexpr auto sycl_global_fence = sycl::access::fence_space::global_space;

// kernel bundle type aliases
using IBundleTy = sycl::kernel_bundle<sycl::bundle_state::input>;
using OBundleTy = sycl::kernel_bundle<sycl::bundle_state::object>;
using EBundleTy = sycl::kernel_bundle<sycl::bundle_state::executable>;

// buffer property aliases
constexpr auto host_ptr = sycl::property::buffer::use_host_ptr{};

// event and profiling aliases
constexpr auto EvtStart = sycl::info::event_profiling::command_start;
constexpr auto EvtEnd = sycl::info::event_profiling::command_end;
constexpr auto EvtStatus = sycl::info::event::command_execution_status;
constexpr auto EvtComplete = sycl::info::event_command_status::complete;

#ifdef _WIN32
// For some reasons, no sycl::atomic_ref in OneAPI Windows release
// Yet it exists in SYCL 2020
// So welcome another hack.
namespace sycl {
template <typename T, memory_order DefaultOrder, memory_scope DefaultScope,
          access::address_space AddressSpace>
using atomic_ref =
    ext::oneapi::atomic_ref<T, DefaultOrder, DefaultScope, AddressSpace>;
}
#endif

// global and local atomic references
template <typename T>
using global_atomic_ref =
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::system,
                     sycl::access::address_space::global_space>;

template <typename T>
using local_atomic_ref =
    sycl::atomic_ref<T, sycl::memory_order::relaxed,
                     sycl::memory_scope::work_group,
                     sycl::access::address_space::local_space>;

// convenient namspaces
namespace esimd = sycl::ext::intel::experimental::esimd;
namespace info = sycl::info;
