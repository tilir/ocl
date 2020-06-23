//-----------------------------------------------------------------------------
//
// framecl helpers: boilerplate and auxiliary stuff
//
//-----------------------------------------------------------------------------

#pragma once

#include <cassert>
#include <iostream>
#include <iterator>

namespace framecl {

// put sequence to ostream
template <typename It>
std::ostream &output(std::ostream &os, It start, It fin) {
  os << "[";
  for (auto it = start, ite = fin; it != ite; ++it) {
    os << *it;
    if (std::next(it) != ite)
      os << " ";
  }
  os << "]";
  return os;
}

// put vector to ostream
template <typename T>
std::ostream &operator<<(std::ostream &os, std::vector<T> v) {
  return output(os, v.begin(), v.end());
}

// generic random-access iterator
// TODO: boost::iterator_facade to reduce boilerplate?
template <typename T> struct randit {
  using iterator = randit;
  using size_type = int;
  using difference_type = ptrdiff_t;
  using value_type = T;
  using reference = T &;
  using pointer = T *;
  using iterator_category = std::random_access_iterator_tag;

private:
  T *pos;
  int inc;

public:
  explicit randit(T *p, int i = 1) : pos(p), inc(i) {
    assert(((inc == 1) || (inc == -1)) && "Illegal increment");
  }
  iterator &operator++() {
    pos += inc;
    return *this;
  }
  iterator &operator--() {
    pos -= inc;
    return *this;
  }
  iterator operator++(int) {
    randit tmp{*this};
    pos += inc;
    return tmp;
  }
  iterator operator--(int) {
    randit tmp{*this};
    pos += inc;
    return tmp;
  }
  iterator &operator-=(size_type n) {
    pos -= (n * inc);
    return *this;
  }
  iterator &operator+=(size_type n) {
    pos += (n * inc);
    return *this;
  }
  iterator operator+(size_type n) {
    randit tmp{*this};
    tmp += n;
    return tmp;
  }
  iterator operator-(size_type n) {
    randit tmp{*this};
    tmp -= n;
    return tmp;
  }
  difference_type operator-(iterator rhs) { return (pos - rhs.pos) * inc; }

public:
  reference operator[](size_type n) const { return pos[n]; }
  reference operator*() { return *pos; }
  pointer operator->() { return pos; }

public:
  bool operator==(randit rhs) { return rhs.pos == pos; }
  bool operator!=(randit rhs) { return rhs.pos != pos; }
  bool operator>(randit rhs) { return rhs.pos > pos; }
  bool operator<(randit rhs) { return rhs.pos < pos; }
  bool operator>=(randit rhs) { return rhs.pos >= pos; }
  bool operator<=(randit rhs) { return rhs.pos <= pos; }
};

} // namespace framecl