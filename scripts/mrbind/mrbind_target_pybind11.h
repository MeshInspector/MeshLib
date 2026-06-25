// Intentionally no pragma once, to allow re-including multiple times with different values of `MB_PB11_STAGE`.

#include <mrbind/targets/pybind11.h>

// A scope guard doesn't include the above header, because this header can be included multiple times, and then `pybind11.h` must also be included multiple times.
#ifndef MR_MRBIND_TARGET_PYBIND11_H_
#define MR_MRBIND_TARGET_PYBIND11_H_

#include "MRMesh/MRExpected.h"
#include "MRMesh/MRSignal.h"
#include "MRPch/MREigen.h"
#include "MRPch/MRJson.h"
#include "MRVoxels/MRVDBFloatGrid.h"

#include <boost/multiprecision/number.hpp>

// Disable all functions accepting some problematic types.

namespace MR::MrbindDetail
{
    template <typename T>
    struct IsEigenMatrix : std::false_type {};
    template<typename Scalar_, int Rows_, int Cols_, int Options_, int MaxRows_, int MaxCols_>
    struct IsEigenMatrix<Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>> : std::true_type {};

    template <typename T>
    struct IsSignal : std::false_type {};
    template <typename T>
    struct IsSignal<Signal<T>> : std::true_type {};

    template <typename T>
    struct IgnoreTypeInBindings : std::bool_constant<
        std::same_as<T, Json::Value> ||
        IsEigenMatrix<T>::value ||
        boost::multiprecision::is_number<T>::value ||
        IsSignal<T>::value
    > {};

    // Recurse into `Expected`.
    template <typename T, typename U>
    struct IgnoreTypeInBindings<Expected<T, U>> : std::bool_constant<IgnoreTypeInBindings<T>::value || IgnoreTypeInBindings<U>::value> {};

    // Recurse into vectors and matrices.
    template <typename T> struct IgnoreTypeInBindings<Vector2<T>> : IgnoreTypeInBindings<T> {};
    template <typename T> struct IgnoreTypeInBindings<Vector3<T>> : IgnoreTypeInBindings<T> {};
    template <typename T> struct IgnoreTypeInBindings<Vector4<T>> : IgnoreTypeInBindings<T> {};
    template <typename T> struct IgnoreTypeInBindings<Matrix2<T>> : IgnoreTypeInBindings<T> {};
    template <typename T> struct IgnoreTypeInBindings<Matrix3<T>> : IgnoreTypeInBindings<T> {};
    template <typename T> struct IgnoreTypeInBindings<Matrix4<T>> : IgnoreTypeInBindings<T> {};

    // Recurse into qualifiers.
    template <typename T> struct IgnoreTypeInBindings<const T> : IgnoreTypeInBindings<T> {};
    template <typename T> struct IgnoreTypeInBindings<T *> : IgnoreTypeInBindings<T> {};
    template <typename T> struct IgnoreTypeInBindings<T &> : IgnoreTypeInBindings<T> {};
    template <typename T> struct IgnoreTypeInBindings<T &&> : IgnoreTypeInBindings<T> {};
    template <typename T> struct IgnoreTypeInBindings<std::shared_ptr<T>> : IgnoreTypeInBindings<T> {};
    template <typename T, typename D> struct IgnoreTypeInBindings<std::unique_ptr<T, D>> : IgnoreTypeInBindings<T> {};
}

template <typename T> requires MR::MrbindDetail::IgnoreTypeInBindings<T>::value
struct MRBind::pb11::IgnoreFuncsWithReturnType<T> : std::true_type {};

template <typename T> requires MR::MrbindDetail::IgnoreTypeInBindings<T>::value
struct MRBind::pb11::ParamTraits<T>
{
    using disables_func = void;
};

template <typename T> requires MR::MrbindDetail::IgnoreTypeInBindings<T>::value
struct MRBind::pb11::IgnoreFieldsWithType<T> : std::true_type {};

// Disable trying to print OpenVDB `FloatGrid`s to fix the following error on Windows:
//     lld-link: error: undefined symbol: class std::basic_ostream<char, struct std::char_traits<char>> & __cdecl openvdb::v11_0::operator<<(class std::basic_ostream<char, struct std::char_traits<char>> &, class openvdb::v11_0::MetaMap const &)
//     >>> referenced by source/TempOutput/PythonBindings/x64/Release/mrmeshpy.fragment.40.o:(public: <auto> __cdecl `void __cdecl MRBind::pb11::TryMakePrintable<struct MR::OpenVdbFloatGrid, class pybind11::class_<struct MR::OpenVdbFloatGrid, class std::shared_ptr<struct MR::OpenVdbFloatGrid>>>(class pybind11::class_<struct MR::OpenVdbFloatGrid, class std::shared_ptr<struct MR::OpenVdbFloatGrid>> &)'::`1'::<lambda_1>::operator()(struct MR::OpenVdbFloatGrid const &) const)
// This is weird, but we don't particularly care about printing the grids.
template <>
struct MRBind::pb11::AllowAutomaticPrinting<MR::OpenVdbFloatGrid> : std::false_type {};


// Custom GIL handling:

// Handle the `MR_BIND_PREFER_UNLOCK_GIL_WHEN_USED_AS_PARAM` macro.
template <typename T> requires requires{typename std::remove_cvref_t<T>::_prefer_gil_unlock_when_used_as_param;}
struct MRBind::pb11::ParamGilHandling<T> : std::integral_constant<MRBind::pb11::GilHandling, MRBind::pb11::GilHandling::prefer_unlock> {};


// Custom `expected` error printing:

// Anything with `toString()`. This is the convention we use.
template <typename T> requires requires(const T &t) {toString(t);}
struct MRBind::pb11::ExpectedErrorToString<T>
{
    std::string operator()(const T &t) const {return toString(t);}
};

#endif
