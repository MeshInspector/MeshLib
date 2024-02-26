#pragma once

#include <concepts>
#include <type_traits>

#include <imgui.h>

// Utilities.
namespace MR::ImGuiMath
{
    namespace detail
    {
        template <typename T>
        concept Scalar = std::is_arithmetic_v<T>;

        template <typename T>
        concept Vector = std::same_as<T, ImVec2> || std::same_as<T, ImVec4>;

        template <typename T>
        concept VectorOrScalar = Vector<T> || Scalar<T>;

        template <typename T>
        concept VectorOrScalarMaybeCvref = VectorOrScalar<std::remove_cvref_t<T>>;

        template <typename T> struct VecSize : std::integral_constant<int, 1> {};
        template <> struct VecSize<ImVec2> : std::integral_constant<int, 2> {};
        template <> struct VecSize<ImVec4> : std::integral_constant<int, 4> {};

        template <int N> struct VecFromSize {};
        template <> struct VecFromSize<2> { using type = ImVec2; };
        template <> struct VecFromSize<4> { using type = ImVec4; };

        // Returns the common vector size between `P...` (cvref-qualifiers are ignored).
        // If there are no vectors in the list, returns `1`.
        // If there are some vectors, returns their size. If the vectors have different sizes, returns -1 to indicate failure.
        template <typename ...P>
        struct CommonVecSize : std::integral_constant<int, 1> {};
        template <typename T, typename ...P>
        struct CommonVecSize<T, P...>
        {
            static constexpr int cur = VecSize<std::remove_cvref_t<T>>::value;
            static constexpr int next = CommonVecSize<P...>::value;
            static constexpr int value =
                cur == 1 || cur == next ? next :
                next == 1 ? cur :
                -1;
        };

        // Whether `P...` are valid operands for a custom operator.
        // All of them must be vectors and scalars (maybe cvref-qualified), and the common vector size must be defined and greater than 1.
        template <typename ...P>
        concept ValidOperands = ( VectorOrScalarMaybeCvref<P> && ... ) && ( CommonVecSize<P...>::value > 1 );
    }

    template <detail::VectorOrScalarMaybeCvref T>
    [[nodiscard]] constexpr auto&& getElem( int i, T&& value )
    {
        if constexpr ( detail::Scalar<std::remove_cvref_t<T>> )
        {
            return value;
        }
        else
        {
            // Technically UB, but helps with optimizations on MSVC for some reason, compared to an if-else chain.
            // GCC and Clang optimize both in the same manner.
            return ( &value.x )[i];
        }
    }

    // Returns a vector, where each element is computed by appling `func()` to individual elements of the `params...`.
    // Scalar parameters are accepted too.
    template <typename F, typename ...P> requires detail::ValidOperands<P...>
    [[nodiscard]] constexpr auto applyElementwise( F&& func, P&&... params ) -> typename detail::VecFromSize<detail::CommonVecSize<P...>::value>::type
    {
        constexpr int n = detail::CommonVecSize<P...>::value;
        typename detail::VecFromSize<n>::type ret;
        for (int i = 0; i < n; i++)
            getElem( i, ret ) = func( getElem( i, params )... );
        return ret;
    }

    // Reduces a vector over a binary `func`.
    template <typename F, detail::VectorOrScalarMaybeCvref T>
    [[nodiscard]] constexpr auto reduce( F&& func, T&& value )
    {
        if constexpr ( std::is_same_v<std::remove_cvref_t<T>, ImVec2> )
            return func( value.x, value.y );
        else if constexpr ( std::is_same_v<std::remove_cvref_t<T>, ImVec2> )
            return func( func( value.x, value.y ), func( value.x, value.w ) );
        else
            return value;
    }
}

// Operators.

template <MR::ImGuiMath::detail::Vector A> [[nodiscard]] constexpr A operator+( A a ) { return a; }
template <MR::ImGuiMath::detail::Vector A> [[nodiscard]] constexpr A operator-( A a ) { return MR::ImGuiMath::applyElementwise( std::negate{}, a ); }

template <typename A, typename B> requires MR::ImGuiMath::detail::ValidOperands<A, B>
[[nodiscard]] constexpr auto operator+( A a, B b ) { return MR::ImGuiMath::applyElementwise( std::plus{}, a, b ); }

template <typename A, typename B> requires MR::ImGuiMath::detail::ValidOperands<A, B>
[[nodiscard]] constexpr auto operator-( A a, B b ) { return MR::ImGuiMath::applyElementwise( std::minus{}, a, b ); }

template <typename A, typename B> requires MR::ImGuiMath::detail::ValidOperands<A, B>
[[nodiscard]] constexpr auto operator*( A a, B b ) { return MR::ImGuiMath::applyElementwise( std::multiplies{}, a, b ); }

template <typename A, typename B> requires MR::ImGuiMath::detail::ValidOperands<A, B>
[[nodiscard]] constexpr auto operator/( A a, B b ) { return MR::ImGuiMath::applyElementwise( std::divides{}, a, b ); }

template <MR::ImGuiMath::detail::Vector A, MR::ImGuiMath::detail::VectorOrScalar B> constexpr A& operator+=( A& a, B b ) { return a = a + b; }
template <MR::ImGuiMath::detail::Vector A, MR::ImGuiMath::detail::VectorOrScalar B> constexpr A& operator-=( A& a, B b ) { return a = a - b; }
template <MR::ImGuiMath::detail::Vector A, MR::ImGuiMath::detail::VectorOrScalar B> constexpr A& operator*=( A& a, B b ) { return a = a * b; }
template <MR::ImGuiMath::detail::Vector A, MR::ImGuiMath::detail::VectorOrScalar B> constexpr A& operator/=( A& a, B b ) { return a = a / b; }

[[nodiscard]] constexpr bool operator==( ImVec2 a, ImVec2 b ) { return a.x == b.x && a.y == b.y; }
[[nodiscard]] constexpr bool operator==( ImVec4 a, ImVec4 b ) { return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w; }
// C++20 generates `!=` from `==`.

namespace MR::ImGuiMath
{
    // Misc functions.

    template <detail::Vector A> [[nodiscard]] constexpr A round( A a ) { return (applyElementwise)( []( auto x ){ return std::round( x ); }, a ); }
    template <detail::Vector A> [[nodiscard]] constexpr A floor( A a ) { return (applyElementwise)( []( auto x ){ return std::floor( x ); }, a ); }
    template <detail::Vector A> [[nodiscard]] constexpr A ceil( A a ) { return (applyElementwise)( []( auto x ){ return std::ceil( x ); }, a ); }

    template <detail::Vector A> [[nodiscard]] constexpr auto dot( A a, A b ) { return reduce( std::plus{}, a * b ); }

    template <detail::Vector A> [[nodiscard]] constexpr auto lengthSq( A a ) { return (dot)( a, a ); }
    template <detail::Vector A> [[nodiscard]] constexpr auto length( A a ) { return std::sqrt( lengthSq( a ) ); }

    template <detail::Vector A> [[nodiscard]] constexpr A normalize( A a ) { auto l = length( a ); return l ? a / l : a; }

    [[nodiscard]] constexpr ImVec2 rot90( ImVec2 a ) { return ImVec2( -a.y, a.x ); }

    template <detail::Vector A, detail::Scalar B> [[nodiscard]] constexpr A mix( B t, A a, A b ) { return a * ( 1 - t ) + b * t; }

    template <typename A, typename B> requires detail::ValidOperands<A, B>
    [[nodiscard]] constexpr auto min( A a, B b ) { return (applyElementwise)( []( auto x, auto y ){ return x < y ? x : y; }, a, b ); }
    template <typename A, typename B> requires detail::ValidOperands<A, B>
    [[nodiscard]] constexpr auto max( A a, B b ) { return (applyElementwise)( []( auto x, auto y ){ return x > y ? x : y; }, a, b ); }

    template <detail::Vector T, typename A, typename B> requires detail::ValidOperands<T, A, B>
    [[nodiscard]] constexpr T clamp( T value, A a, B b ) { return (max)( (min)( value, b ), a ); }

    // Comparison helpers.

    // Usage: `Compare{Any,All}( vector ) @ vector_or_scalar`.
    // For example: `CompareAll( vector ) >= 42`.

    template <typename Derived, detail::VectorOrScalar A, bool All>
    struct BasicVectorCompareHelper
    {
        A value;

        constexpr BasicVectorCompareHelper( A value ) : value( value ) {}

        template <typename F, detail::ValidOperands<A> B>
        [[nodiscard]] constexpr bool compare( F&& func, B other ) const
        {
            for ( int i = 0; i < detail::VecSize<A>::value; i++ )
            {
                if ( func( getElem( i, value ), getElem( i, other ) ) != All )
                    return !All;
            }
            return All;
        }

        template <detail::ValidOperands<A> B> [[nodiscard]] constexpr bool operator==( B other ) const { return compare( std::equal_to{}, other ); }
        template <detail::ValidOperands<A> B> [[nodiscard]] constexpr bool operator!=( B other ) const { return compare( std::not_equal_to{}, other ); }
        template <detail::ValidOperands<A> B> [[nodiscard]] constexpr bool operator< ( B other ) const { return compare( std::less{}, other ); }
        template <detail::ValidOperands<A> B> [[nodiscard]] constexpr bool operator> ( B other ) const { return compare( std::greater{}, other ); }
        template <detail::ValidOperands<A> B> [[nodiscard]] constexpr bool operator<=( B other ) const { return compare( std::less_equal{}, other ); }
        template <detail::ValidOperands<A> B> [[nodiscard]] constexpr bool operator>=( B other ) const { return compare( std::greater_equal{}, other ); }
    };

    template <detail::VectorOrScalar A>
    struct CompareAll : BasicVectorCompareHelper<CompareAll<A>, A, true>
    {
        using BasicVectorCompareHelper<CompareAll<A>, A, true>::BasicVectorCompareHelper;
    };
    template <detail::VectorOrScalar A>
    CompareAll( A ) -> CompareAll<A>;

    template <detail::VectorOrScalar A>
    struct CompareAny : BasicVectorCompareHelper<CompareAll<A>, A, false>
    {
        using BasicVectorCompareHelper<CompareAll<A>, A, false>::BasicVectorCompareHelper;
    };
    template <detail::VectorOrScalar A>
    CompareAny( A ) -> CompareAny<A>;
}
