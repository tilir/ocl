//-----------------------------------------------------------------------------
//
// framecl helpers: boilerplate and auxiliary stuff
//
//-----------------------------------------------------------------------------

#pragma once

#include <cassert>
#include <iostream>
#include <iterator>
#include <random>

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

// generic random initialization
template <typename T> void rand_init(T &arr, int sz, int low, int up) {
  static std::mt19937_64 mt_source;
  std::uniform_int_distribution<int> dist(low, up);
  for (int i = 0; i < sz; ++i)
    arr[i] = dist(mt_source);
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
  reference operator[](size_type n) const { return *(pos + n * inc); }
  reference operator*() { return *pos; }
  pointer operator->() { return pos; }

public:
  bool operator==(randit rhs) { return rhs.pos == pos; }
  bool operator!=(randit rhs) { return rhs.pos != pos; }
  bool operator>(randit rhs) { return (*this - rhs) > 0; }
  bool operator>=(randit rhs) { return (*this > rhs) || (*this == rhs); }
  bool operator<(randit rhs) { return !(*this >= rhs); }
  bool operator<=(randit rhs) { return !(*this > rhs); }
};

} // namespace framecl