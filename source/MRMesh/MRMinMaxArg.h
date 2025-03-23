#pragma once

#include <limits>

namespace MR
{

/// minimum and maximum of some vector with values of type T,
/// and the indices of where minimum and maximum are reached of type I
template<typename T, typename I>
struct MinMaxArg
{
    T min = std::numeric_limits<T>::max();
    T max = std::numeric_limits<T>::lowest();
    I minArg, maxArg;
};

} //namespace MR
