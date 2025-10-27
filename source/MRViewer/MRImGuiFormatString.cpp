#include "MRImGuiFormatString.h"
#include "MRMesh/MRString.h"

namespace MR
{

template <UnitEnum E, detail::Units::Scalar T>
std::string valueToImGuiFormatString( T value, const UnitToStringParams<E>& params )
{
    std::string ret = replace( valueToString( value, params ), "%", "%%" );
    ret += "##%";

    if constexpr ( std::is_integral_v<T> )
    {
        using SignedT = std::make_signed_t<T>;
        if constexpr ( std::is_same_v<SignedT, signed char> )
            ret += "hh";
        else if constexpr ( std::is_same_v<SignedT, short> )
            ret += "h";
        else if constexpr ( std::is_same_v<SignedT, int> )
            ret += "";
        else if constexpr ( std::is_same_v<SignedT, long> )
            ret += "l";
        else if constexpr ( std::is_same_v<SignedT, long long> )
            ret += "ll";
        else
            static_assert( dependent_false<SignedT>, "Unknown integral type." );

        ret += std::is_signed_v<T> ? "d" : "u";
    }
    else
    {
        int precision = 0;
        std::size_t pos = ret.find( '.' );
        if ( pos != std::string::npos )
        {
            pos++;
            while (
                std::isdigit( (unsigned char)ret[pos + precision] ) ||
                ( params.thousandsSeparatorFrac && ret[pos + precision] == params.thousandsSeparatorFrac )
            )
                precision++;
        }

        fmt::format_to( std::back_inserter( ret ), ".{}", precision );

        if constexpr ( std::is_same_v<T, float> )
            ; // Nothing.
        else if constexpr ( std::is_same_v<T, double> )
            ; // Nothing. Same as for `float`, yes.
        else if constexpr ( std::is_same_v<T, long double> )
            ret += 'L';
        else
            static_assert( dependent_false<T>, "Unknown floating-point type." );

        if ( params.style == NumberStyle::exponential )
            ret += 'e';
        else if ( params.style == NumberStyle::maybeExponential )
            ret += 'g';
        else
            ret += 'f';
    }

    return ret;
}

#define MR_Y(T, E) template std::string valueToImGuiFormatString( T value, const UnitToStringParams<E>& params );
#define MR_X(E) DETAIL_MR_UNIT_VALUE_TYPES(MR_Y, E)
DETAIL_MR_UNIT_ENUMS(MR_X)
#undef MR_X
#undef MR_Y

template <detail::Units::Scalar T>
std::string valueToImGuiFormatString( T value, const VarUnitToStringParams& params )
{
    return std::visit( [&]( const auto& visitedParams )
    {
        return (valueToImGuiFormatString)( value, visitedParams );
    }, params );
}

#define MR_X(T, unused) template std::string valueToImGuiFormatString<T>( T value, const VarUnitToStringParams& params );
DETAIL_MR_UNIT_VALUE_TYPES(MR_X,)
#undef MR_X

} //namespace MR
