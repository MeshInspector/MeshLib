// Intentionally no pragma once, though currently it doesn't matter (see the mrbind header included below).

#include <mrbind/targets/pybind11.h>

// A scope guard doesn't include the above header, because this header can be included multiple times, and then `pybind11.h` must also be included multiple times.
#ifndef MR_MRBIND_TARGET_PYBIND11_H_
#define MR_MRBIND_TARGET_PYBIND11_H_

#include "MRPch/MRJson.h"

// Disable all functions accepting or returning `Json::Value`.

template <typename T> requires std::is_same_v<typename MRBind::pb11::StripToUnderlyingType<T>::type, Json::Value>
struct MRBind::pb11::ReturnTypeTraits<T>
{
    using no_adjust = void;
    using disables_func = void;
};

template <typename T> requires std::is_same_v<typename MRBind::pb11::StripToUnderlyingType<T>::type, Json::Value>
struct MRBind::pb11::ParamTraits<T>
{
    using disables_func = void;
};

#endif
