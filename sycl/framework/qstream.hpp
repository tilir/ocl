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

#include <iostream>

namespace sycltesters {

enum class QSState { Quiet = 0, Loud = 1 };

class QuietStream {
  QSState State_;

public:
  QuietStream(QSState State = QSState::Loud) : State_(State) {}
  QSState state() const { return State_; }
  QSState silence(bool Quiet) {
    QSState Old = State_;
    State_ = Quiet ? QSState::Quiet : QSState::Loud;
    return Old;
  }
};

inline QuietStream qout;

template <typename T> QuietStream &operator<<(QuietStream &Qstream, T &&data) {
  if (Qstream.state() == QSState::Loud)
    std::cout << std::forward<T>(data);
  return Qstream;
}

} // namespace sycltesters