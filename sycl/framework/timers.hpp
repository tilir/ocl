//------------------------------------------------------------------------------
//
// Generic code for timer and event timing
//
//------------------------------------------------------------------------------
//
// This file is licensed after LGPL v3
// Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
//
//------------------------------------------------------------------------------

#pragma once

#include <CL/sycl.hpp>

#include "syclconst.hpp"

namespace chrono = std::chrono;

// milliseconds, microsecond and nanoseconds
static const double msec_per_sec = 1000.0;
static const double usec_per_sec = msec_per_sec * msec_per_sec;
static const double nsec_per_sec = msec_per_sec * msec_per_sec * msec_per_sec;

namespace sycltesters {

class Timer {
  chrono::high_resolution_clock::time_point Start, Fin;
  bool Started = false;

public:
  Timer() = default;
  void start() {
    assert(!Started);
    Started = true;
    Start = chrono::high_resolution_clock::now();
  }
  void stop() {
    assert(Started);
    Started = false;
    Fin = chrono::high_resolution_clock::now();
  }
  unsigned elapsed() {
    assert(!Started);
    auto Elps = Fin - Start;
    auto Msec = chrono::duration_cast<chrono::milliseconds>(Elps);
    return Msec.count();
  }
};

struct NamedEvent {
  sycl::event Evt_;
  std::string Name_;
  NamedEvent(sycl::event Evt, std::string Name = "Unnamed")
      : Evt_(Evt), Name_(std::move(Name)) {}
};

using EvtVec_t = std::vector<NamedEvent>;
using EvtRet_t = std::optional<EvtVec_t>;

inline unsigned long long getTime(EvtRet_t Opt, bool Quiet = true) {
  unsigned long long AccTime = 0;
  int EvtIdx = 0;
  if (!Opt.has_value())
    return AccTime;
  auto &&Evts = Opt.value();
  auto Old = qout.set(Quiet);
  for (auto &&NEvt : Evts) {
    sycl::event &Evt = NEvt.Evt_;
    qout << EvtIdx++ << " (" << NEvt.Name_ << "): ";
    sycl::info::event_command_status EStatus =
        Evt.template get_info<EvtStatus>();
    if (EStatus != EvtComplete) {
      qout << " [...] ";
      Evt.wait();
    }
    auto Start = Evt.template get_profiling_info<EvtStart>();
    auto End = Evt.template get_profiling_info<EvtEnd>();
    auto Elapsed = End - Start;
    AccTime += Elapsed;
    qout << Elapsed / nsec_per_sec << " : " << AccTime / nsec_per_sec << "\n";
  }
  qout.set(Old);
  return AccTime;
}

} // namespace sycltesters
