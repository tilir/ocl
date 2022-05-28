//------------------------------------------------------------------------------
//
// Quietable stream
//
//------------------------------------------------------------------------------
//
// This file is licensed after LGPL v3
// Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
//
//------------------------------------------------------------------------------

#pragma once

#include <atomic>
#include <iostream>

namespace sycltesters {

enum class QSState { Quiet = 0, Loud = 1 };

// we need to have MT guarantees, because qout is shared global object
class QuietStream : private std::streambuf, public std::ostream {
  std::atomic<QSState> State_;

public:
  QuietStream(QSState State = QSState::Loud)
      : std::ostream(this), State_(State) {}
  QSState state() const { return State_.load(); }
  QSState set(QSState State) {
    QSState Old = State_.exchange(State);
    return Old;
  }

  // handy overload for true / false
  QSState set(bool Quiet) {
    return set(Quiet ? QSState::Quiet : QSState::Loud);
  }

private:
  int overflow(int c) override {
    if (State_.load() == QSState::Loud) {
      std::cout.put(c);
      return 0;
    }
    return c;
  }
};

inline QuietStream qout;

} // namespace sycltesters