//-----------------------------------------------------------------------------
//
// framecl core: context, device, etc
//
//-----------------------------------------------------------------------------
//
// task system:
//   ITask base class defines interface:
//      execute(command_queue, events)
//   Every concrete task is ITask subclass:
//     BufferReadTask
//     BufferWriteTask
//     ProgramExecTask
//     ....
//   task_t class keeps unique_pointer to ITask and knows how to enqueue:
//     task_t::enqueue_on(command_queue, events)
//       calls ITask::execute(command_queue, events)
//   depgraph_t manages task_t's and events
//     depgraph_t::execute()
//       calls context_t::enqueue(task_t, events)
//   context_t owns queue and delegates to
//     context_t::enqueue(task_t, events)
//       calls task_t::enqueue_on(command_queue, events)
//
// buffer types:
//   owning with host buffer and new buffer object
//   non-owning with host buffer and new buffer object
//   non-owning with host buffer and pre-existing buffer object (dangerous!)
//
//-----------------------------------------------------------------------------
//
// Wrapped against official Khronos C++ bindings:
//   http://github.khronos.org/OpenCL-CLHPP/
//
//-----------------------------------------------------------------------------

#pragma once

#include <cassert>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>

// clang-format off
#include "cldefs.h"
#include "CL/opencl.hpp"
// clang-format on

#include "fchelper.hpp"
#include "fcplatform.hpp"
#include "fcutils.hpp"

namespace chrono = std::chrono;

namespace framecl {

// output and input events
struct eventguards_t {
  cl::Event out;
  std::vector<cl::Event> ins;
};

// group of devices on single platform with shared context
// each have individual queue
template <typename Task> class devgroup_t final {
  cl::Context ctx_;
  std::vector<cl::Device> devices_;
  std::vector<cl::CommandQueue> queues_;
  int active_ = 0;

public:
  using task_t = Task;

  // this ctor means that program is file in opts.program
  devgroup_t(optparser_t opts) {
    assert(opts.parsed() && "You can not create devgroup before parsing opts");
    platform_list_t platforms;
    auto p = platforms.select(opts.platform());
    p.getDevices(CL_DEVICE_TYPE_ALL, &devices_);
    assert(devices_.size() > 0 && "Failed to query devices");
    ctx_ = cl::Context(devices_);
    queues_.resize(devices_.size());
    for (int i = 0, sz = queues_.size(); i < sz; ++i)
      queues_[i] = cl::CommandQueue(ctx_, devices_[i],
                                    cl::QueueProperties::Profiling |
                                        cl::QueueProperties::OutOfOrder);
  }

  int size() const { return devices_.size(); }

  // chooses to which device enqueue
  void select(int n) {
    assert(n >= 0 && n < size() &&
           "can not select device: number incorrect, check size()");
    active_ = n;
  }

  int global_mem_size() const {
    return devices_[active_].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
  }

  int max_workgroup_size() const {
    return devices_[active_].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  }

  void enqueue(Task *pt, eventguards_t &evts) const {
    pt->enqueue_on(queues_[active_], evts);
  }

  operator cl::Context() const { return ctx_; }
};

enum class task { write, process, read, map, unmap, native };

struct ITask {
  virtual void execute(cl::CommandQueue q, eventguards_t &evts) = 0;
  virtual ~ITask() {}
};

class task_t {
  task type_;
  std::unique_ptr<ITask> tsk_;

public:
  template <typename T, typename... Args>
  task_t(task type, T &obj, Args &&... args)
      : type_{type}, tsk_{obj.get_task(type, std::forward<Args>(args)...)} {}

  void enqueue_on(cl::CommandQueue q, eventguards_t &evts) const {
    assert(tsk_ && "Non-null task expected");
    tsk_->execute(q, evts);
  }

  void dump(std::ostream &os) const {
    os << "[task: " << tsk_.get() << " ";
    switch (type_) {
    case task::write:
      os << "write";
      break;
    case task::process:
      os << "process";
      break;
    case task::read:
      os << "read";
      break;
    default:
      throw std::runtime_error("unknown task type");
    }
    os << "]";
  }
};

using context_t = devgroup_t<task_t>;

class program_t final {
  context_t &ctx_;
  cl::Program prog_;
  std::string pline_;

  void build_program() {
    prog_ = cl::Program(ctx_, pline_);
    try {
      prog_.build(); // "-cl-std=CL2.0" ?
    } catch (...) {
      auto buildInfo = prog_.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
      std::cerr << "Error: build failed. Log:" << std::endl;
      for (auto &&pair : buildInfo)
        std::cerr << pair.second << std::endl << std::endl;
      throw;
    }
  }

  void load_program(std::string name) {
    std::ifstream is;
    is.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    is.open(name);
    std::stringstream os;
    os << is.rdbuf();
    pline_ = os.str();
  }

public:
  program_t(context_t &ctx, optparser_t opts) : ctx_{ctx} {
    assert(opts.parsed() && "You can not create devgroup before parsing opts");
    load_program(opts.program());
    build_program();
  }

  using add_text_functor_t = std::function<void(std::string &)>;

  program_t(context_t &ctx, optparser_t opts, add_text_functor_t fn)
      : ctx_{ctx} {
    assert(opts.parsed() && "You can not create devgroup before parsing opts");
    load_program(opts.program());
    fn(pline_);
    build_program();
  }

  // UGLY breach of incapsulation
  context_t &context() { return ctx_; }

  operator cl::Program() const { return prog_; }
};

template <typename F> class native_t final {
  context_t &ctx_;
  F closure_;

  static void CL_CALLBACK do_call(void *closure) {
    auto callable = *reinterpret_cast<F *>(closure);
    callable();
  }

public:
  native_t(F closure) : closure_(closure) {}

  std::unique_ptr<ITask> get_task(task type);
};

struct run_params_t {
  cl::NDRange offset, global, local;
};

template <typename... Ts> class functor_t final {
  context_t &ctx_;
  run_params_t rp_;
  cl::KernelFunctor<Ts...> kf_;

public:
  functor_t(program_t prog, run_params_t rp, std::string kname)
      : ctx_(prog.context()), rp_(rp), kf_(prog, kname) {}

  template <typename... Vs>
  std::unique_ptr<ITask> get_task(task type, Vs &&... args);
};

// simple buffer on host registered on device ctx
template <typename T> class buffer_t final {
  context_t &ctx_;
  int size_;
  T *contents_;
  cl::Buffer buf_;
  bool owning = false;

  // core buffer ctors and interface
public:
  buffer_t(context_t &ctx, int size)
      : ctx_(ctx), size_(size), contents_(new T[size_]),
        buf_(cl::Buffer(ctx, CL_MEM_READ_WRITE, size * sizeof(T))),
        owning(true) {}

  buffer_t(context_t &ctx, T *contents, int size)
      : ctx_(ctx), size_(size), contents_(contents),
        buf_(cl::Buffer(ctx, CL_MEM_READ_WRITE, size * sizeof(T))) {}

  buffer_t(context_t &ctx, cl::Buffer b, T *contents, int size)
      : ctx_(ctx), size_(size), contents_(contents), buf_(b) {}

  buffer_t(const buffer_t &) = delete;
  buffer_t &operator=(const buffer_t &) = delete;

  buffer_t(buffer_t &&rhs)
      : ctx_(rhs.ctx_), size_(rhs.size_), contents_(rhs.contents_),
        buf_(rhs.buf_) {
    rhs.contents_ = nullptr;
    rhs.buf_ = cl::Buffer();
  }

  buffer_t &operator=(buffer_t &&) = delete;

  ~buffer_t() {
    if (owning)
      delete[] contents_;
  }

  std::unique_ptr<ITask> get_task(task type);

  // mimicing vector-like interface
public:
  auto begin() { return randit(&contents_[0]); }
  auto end() { return randit(&contents_[0] + size_); }
  auto rbegin() { return randit(&contents_[size_ - 1], -1); }
  auto rend() { return randit(&contents_[0] - 1, -1); }

  int size() const noexcept { return size_; }
  T &operator[](int n) { return contents_[n]; }

  cl::Buffer base() const { return buf_; }

  void dump(std::ostream &os) { output(os, contents_, contents_ + size_); }
};

struct BufferReadTask : public ITask {
  int off_, size_;
  void *ptr_;
  cl::Buffer buf_;

  BufferReadTask(cl::Buffer buf, int off, int sz, void *ptr)
      : off_(off), size_(sz), ptr_(ptr), buf_(buf) {}

  void execute(cl::CommandQueue q, eventguards_t &evts) override {
    q.enqueueReadBuffer(buf_, /* bloking */ false, off_, size_, ptr_, &evts.ins,
                        &evts.out);
  }
};

struct BufferWriteTask : public ITask {
  int off_, size_;
  void *ptr_;
  cl::Buffer buf_;

  BufferWriteTask(cl::Buffer buf, int off, int sz, void *ptr)
      : off_(off), size_(sz), ptr_(ptr), buf_(buf) {}

  void execute(cl::CommandQueue q, eventguards_t &evts) override {
    q.enqueueWriteBuffer(buf_, /* bloking */ false, off_, size_, ptr_,
                         &evts.ins, &evts.out);
  }
};

struct BufferMapTask : public ITask {
  cl::Buffer buf_;
  cl_map_flags flags_;
  int off_, size_;
  void *&ptr_;

  BufferMapTask(cl::Buffer buf, cl_map_flags flags, int off, int sz, void *&ptr)
      : buf_(buf), flags_(flags), off_(off), size_(sz), ptr_(ptr) {}

  void execute(cl::CommandQueue q, eventguards_t &evts) override {
    ptr_ = q.enqueueMapBuffer(buf_, /* bloking */ false, flags_, off_, size_,
                              &evts.ins, &evts.out);
  }
};

struct BufferUnMapTask : public ITask {
  cl::Memory buf_;
  void *ptr_;

  BufferUnMapTask(cl::Memory buf, void *ptr) : buf_(buf), ptr_(ptr) {}

  void execute(cl::CommandQueue q, eventguards_t &evts) override {
    q.enqueueUnmapMemObject(buf_, ptr_, &evts.ins, &evts.out);
  }
};

struct NativeKernelTask : public ITask {
  using callback_ft = void(CL_CALLBACK *)(void *);

  callback_ft cb_;
  std::pair<void *, cl::size_type> args_;

  NativeKernelTask(callback_ft cb, std::pair<void *, cl::size_type> args)
      : cb_(cb), args_(args) {}

  void execute(cl::CommandQueue q, eventguards_t &evts) override {
    q.enqueueNativeKernel(cb_, args_, /* mem objs */ nullptr,
                          /* host objs */ nullptr, &evts.ins, &evts.out);
  }
};

template <typename F, typename... Args> struct ProgramExecTask : public ITask {
  run_params_t rp_;
  F func_;
  std::function<cl::Event(cl::EnqueueArgs)> f_;

  // not perfect forwarding because lambda will be used out of scope, so I don't
  // want to mess with dangling stuff
  // TODO: rethink it later
  ProgramExecTask(run_params_t rp, F func, Args... args)
      : rp_(rp), func_(func) {
    auto lam = [=, this](cl::EnqueueArgs eargs) mutable {
      return func_(eargs, std::forward<Args>(args)...);
    };
    f_ = lam;
  }

  void execute(cl::CommandQueue q, eventguards_t &evts) override {
    cl::EnqueueArgs eargs(q, evts.ins, rp_.offset, rp_.global, rp_.local);
    evts.out = f_(eargs);
  }
};

template <typename F> std::unique_ptr<ITask> native_t<F>::get_task(task type) {
  if (type == task::native) {
    std::pair<void *, cl::size_type> args{&closure_, sizeof(closure_)};
    return std::unique_ptr<ITask>{new NativeKernelTask(do_call, args)};
  }
}

template <typename T> std::unique_ptr<ITask> buffer_t<T>::get_task(task type) {
  if (type == task::read)
    return std::unique_ptr<ITask>{
        new BufferReadTask(buf_, 0, size_ * sizeof(T), &contents_[0])};
  if (type == task::write)
    return std::unique_ptr<ITask>{
        new BufferWriteTask(buf_, 0, size_ * sizeof(T), &contents_[0])};
  if (type == task::map) {
    void *pcontents = nullptr;
    assert(contents_ == nullptr &&
           "Mapped buffer contents must be null before mapping");
    auto task = std::unique_ptr<ITask>{new BufferMapTask(
        buf_, CL_MAP_READ | CL_MAP_WRITE, 0, size_ * sizeof(T), pcontents)};
    contents_ = reinterpret_cast<T *>(pcontents);
    return task;
  }

  if (type == task::unmap) {
    assert(contents_ != nullptr &&
           "Unmapped buffer contents must be not null before unmapping");
    void *pcontents = contents_;
    return std::unique_ptr<ITask>{new BufferUnMapTask(buf_, pcontents)};
    contents_ = nullptr;
  }

  throw std::runtime_error("Illegal task for buffer");
}

template <typename... Ts>
template <typename... Vs>
std::unique_ptr<ITask> functor_t<Ts...>::get_task(task type, Vs &&... args) {
  if (type == task::process)
    return std::unique_ptr<ITask>{
        new ProgramExecTask(rp_, kf_, std::forward<Vs>(args)...)};
  throw std::runtime_error("Illegal task for program");
}

class depgraph_t {
  context_t &ctx_;
  std::vector<task_t *> tasks_;
  std::vector<cl::Event> evts_;
  std::vector<cl_ulong> evt_times_;
  std::unordered_map<task_t *, int> idx_;
  std::unordered_map<task_t *, std::vector<task_t *>> deps_;
  std::unordered_map<task_t *, int> task_levels_;
  std::vector<std::vector<task_t *>> level_tasks_;
  chrono::high_resolution_clock::time_point tstart_, tfin_;
  bool executed_ = false;

  template <typename It> void add_tasks(It start, It fin);

public:
  // initialize if tasks will be added from some external container
  // like vector of vectors
  template <typename It>
  depgraph_t(context_t &ctx, It start, It fin) : ctx_(ctx) {
    add_tasks(start, fin);
  }

  // initialize adding tasks from initializer list to make simple examples
  // pretty
  depgraph_t(context_t &ctx,
             std::initializer_list<std::initializer_list<task_t *>> init)
      : ctx_(ctx) {
    add_tasks(init.begin(), init.end());
  }

  void dump(std::ostream &os) const {
    int nlv = 0;
    for (auto &&lv : level_tasks_) {
      os << (nlv++) << ": ";
      for (auto &&tsk : lv) {
        tsk->dump(os);
        os << " ";
      }
      os << "\n";
    }
  }

  void execute(bool evtdbg = false);

  unsigned long elapsed() const {
    assert(executed_ && "You need execute before measuring time");
    return chrono::duration_cast<chrono::milliseconds>(tfin_ - tstart_).count();
  }

  cl_ulong task_elapsed(task_t *pt) {
    assert(executed_ && "You need execute before measuring time");
    auto idit = idx_.find(pt);
    assert(idx_.end() != idit && "You querying non-existent tasks");
    int idx = idit->second;
    return evt_times_[idx];
  }

  const char *cstat(int status) {
    switch (status) {
    case CL_QUEUED:
      return "Queued";
    case CL_SUBMITTED:
      return "Submitted";
    case CL_RUNNING:
      return "Running";
    case CL_COMPLETE:
      return "Complete";
    default:
      throw std::runtime_error("Unknown status of event");
    }
  }
};

template <typename It> void depgraph_t::add_tasks(It start, It fin) {
  // fill tasks, idx and deps
  level_tasks_.emplace_back();
  for (auto it = start; it != fin; ++it) {
    auto ls = *it;
    assert(ls.size() > 0 && "Void rows not acceptable");
    task_t *pt = *ls.begin();
    tasks_.push_back(pt);
    idx_[pt] = tasks_.size() - 1;

    // fill level 0 if no deps
    task_levels_[pt] = (ls.size() == 1) ? 0 : -1;
    if (ls.size() == 1)
      level_tasks_[0].push_back(pt);
    else {
      // fill deps
      for (auto &&it = std::next(ls.begin()), ite = ls.end(); it != ite; ++it)
        deps_[pt].push_back(*it);
    }
  }
  evts_.resize(tasks_.size());
  evt_times_.resize(tasks_.size());

  // fill levels: any pass of this loop will form next level
  for (;;) {
    bool all_set = true;
    bool levels_modified = false;
    for (auto &&pt : tasks_) {
      if (task_levels_[pt] != -1)
        continue;
      all_set = false;
      int level = -1;
      for (auto &&pdep : deps_[pt]) {
        int dep_lv = task_levels_[pdep];
        if (dep_lv == -1) {
          level = -1;
          break;
        }
        level = std::max(level, dep_lv);
      }
      if (level != -1) {
        levels_modified = true;
        task_levels_[pt] = level + 1;
        int tsz = level_tasks_.size();
        assert(tsz >= 0 && "Too many tasks");
        assert(level + 1 <= tsz && "This invariant shall hold by definition");
        if (level + 1 == tsz)
          level_tasks_.emplace_back();
        level_tasks_[level + 1].push_back(pt);
      }
    }

    if (all_set)
      break;

    if (!levels_modified)
      throw std::runtime_error("incorrect dep-graph structure detected");
  }
}

// time measure will measure nothing if evtdbg is on
void depgraph_t::execute(bool evtdbg) {
  std::vector<bool> last_active_;
  tstart_ = chrono::high_resolution_clock::now();

  // peek task
  for (auto &&lv : level_tasks_) {
    std::vector<bool> active_(evts_.size(), false);
    for (auto &&pt : lv) {
      int id = idx_[pt];
      eventguards_t evt;
      for (auto &&pdep : deps_[pt]) {
        int dep_id = idx_[pdep];
        if (evtdbg) {
          std::cout << "Dependency event: " << dep_id << "; ";
          int status =
              evts_[dep_id].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
          std::cout << "Status: " << cstat(status) << "\n";
        }
        evt.ins.push_back(evts_[dep_id]);
      }
      ctx_.enqueue(pt, evt);
      evts_[id] = evt.out;
      if (evtdbg) {
        std::cout << "Produced event: " << id << "; ";
        int status = evts_[id].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
        std::cout << "Status: " << cstat(status) << "\n";
      }
      active_[id] = true;
    }
    last_active_.swap(active_);
  }

  // wait for last active events after all is about to finish
  for (size_t i = 0; i < last_active_.size(); ++i)
    if (last_active_[i]) {
      evts_[i].wait();
      if (evtdbg) {
        std::cout << "Active event: " << i << "; ";
        int status = evts_[i].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
        std::cout << "Status: " << cstat(status) << "\n";
      }
    }

  tfin_ = chrono::high_resolution_clock::now();

  // get profiling information for all events
  for (int idx = 0, idf = evts_.size(); idx < idf; ++idx) {
    auto &event = evts_[idx];
    cl_ulong time_start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong time_end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    evt_times_[idx] = time_end - time_start;
  }

  executed_ = true;
}

} // namespace framecl
