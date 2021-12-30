#pragma once
#include "MRVector2.h"
#include "MRVector3.h"
#include "MRVector4.h"
#include "MRMatrix2.h"
#include "MRMatrix3.h"
#include "MRMatrix4.h"
#include "MRAffineXf.h"
#include "MRId.h"
#include "MRColor.h"

#include <type_traits>

namespace std
{

template<size_t I, typename T>
struct tuple_element<I, MR::Vector2<T>> { using type = typename MR::Vector2<T>::ValueType; };

template<size_t I, typename T>
struct tuple_element<I, MR::Vector3<T>> { using type = typename MR::Vector3<T>::ValueType; };

template<size_t I, typename T>
struct tuple_element<I, MR::Vector4<T>> { using type = typename MR::Vector4<T>::ValueType; };

template<size_t I, typename T>
struct tuple_element<I, MR::Matrix2<T>> { using type = typename MR::Matrix2<T>::VectorType; };

template<size_t I, typename T>
struct tuple_element<I, MR::Matrix3<T>> { using type = typename MR::Matrix3<T>::VectorType; };

template<size_t I, typename T>
struct tuple_element<I, MR::Matrix4<T>> { using type = typename MR::Matrix4<T>::VectorType; };

template<typename V>
struct tuple_element<0, MR::AffineXf<V>> { using type = typename V::MatrixType; };

template<typename V>
struct tuple_element<1, MR::AffineXf<V>> { using type = V; };

template <typename T> 
struct tuple_element<0, MR::Id<T>> { using type = int; };

template<size_t I>
struct tuple_element<I, MR::Color> { using type = uint8_t; };

template<typename T>
struct tuple_size<MR::Vector2<T>> : integral_constant<size_t, MR::Vector2<T>::elements> {};

template<typename T>
struct tuple_size<MR::Vector3<T>> : integral_constant<size_t, MR::Vector3<T>::elements> {};

template<typename T>
struct tuple_size<MR::Vector4<T>> : integral_constant<size_t, MR::Vector4<T>::elements> {};

template<typename T>
struct tuple_size<MR::Matrix2<T>> : integral_constant<size_t, MR::Matrix2<T>::VectorType::elements> {}; // as far as matrix as square num vector elements is equal to num vectors

template<typename T>
struct tuple_size<MR::Matrix3<T>> : integral_constant<size_t, MR::Matrix3<T>::VectorType::elements> {}; // as far as matrix as square num vector elements is equal to num vectors

template<typename T>
struct tuple_size<MR::Matrix4<T>> : integral_constant<size_t, MR::Matrix4<T>::VectorType::elements> {}; // as far as matrix as square num vector elements is equal to num vectors

template<typename V>
struct tuple_size<MR::AffineXf<V>> : integral_constant<size_t, 2> {}; // 2 here - matrix and translation

template <typename T> 
struct tuple_size<MR::Id<T>> : integral_constant<size_t, 1> {};

template <>
struct tuple_size<MR::Color> : integral_constant<size_t, 4> {};
}

namespace MR
{

template<size_t I, typename T>
constexpr const T& get( const Vector2<T>& v ) noexcept { return v[int( I )]; }

template<size_t I, typename T>
constexpr T& get( Vector2<T>& v ) noexcept { return v[int( I )]; }

template<size_t I, typename T>
constexpr const T& get( const Vector3<T>& v ) noexcept { return v[int( I )]; }

template<size_t I, typename T>
constexpr T& get( Vector3<T>& v ) noexcept { return v[int( I )]; }

template<size_t I, typename T>
constexpr const T& get( const Vector4<T>& v ) noexcept { return v[int( I )]; }

template<size_t I, typename T>
constexpr T& get( Vector4<T>& v ) noexcept { return v[int( I )]; }

template<size_t I, typename T>
constexpr const typename Matrix2<T>::VectorType& get( const Matrix2<T>& m ) noexcept { return m[int( I )]; }

template<size_t I, typename T>
constexpr typename Matrix2<T>::VectorType& get( Matrix2<T>& m ) noexcept { return m[int( I )]; }

template<size_t I, typename T>
constexpr const typename Matrix3<T>::VectorType& get( const Matrix3<T>& m ) noexcept { return m[int( I )]; }

template<size_t I, typename T>
constexpr typename Matrix3<T>::VectorType& get( Matrix3<T>& m ) noexcept { return m[int( I )]; }

template<size_t I, typename T>
constexpr const typename Matrix4<T>::VectorType& get( const Matrix4<T>& m ) noexcept { return m[int( I )]; }

template<size_t I, typename T>
constexpr typename Matrix4<T>::VectorType& get( Matrix4<T>& m ) noexcept { return m[int( I )]; }

template<size_t I, typename V>
constexpr const typename std::tuple_element<I, AffineXf<V>>::type& get( const AffineXf<V>& m ) noexcept
{
    if constexpr ( I == 0 )
        return m.A;
    else
        return m.b;
}

template<size_t I, typename V>
constexpr typename std::tuple_element<I, AffineXf<V>>::type& get( AffineXf<V>& m ) noexcept
{
    if constexpr ( I == 0 )
        return m.A;
    else
        return m.b;
}

template <size_t I, typename T> 
constexpr int get( const MR::Id<T> & id ) noexcept
{
    static_assert( I == 0 );
    return (int)id;
}

template <size_t I, typename T> 
constexpr int & get( MR::Id<T>& id ) noexcept
{
    static_assert( I == 0 );
    return id.get();
}

template<size_t I>
constexpr const uint8_t& get( const Color& c ) noexcept
{
    static_assert( I < 4 );
    return c[int( I )];
}

template<size_t I>
constexpr uint8_t& get( Color& c ) noexcept
{
    static_assert( I < 4 );
    return c[int( I )];
}

} //namespace MR
