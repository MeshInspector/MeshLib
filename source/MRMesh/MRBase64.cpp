#include "MRBase64.h"
#include "MRTimer.h"
#include <boost/archive/iterators/binary_from_base64.hpp>
#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/transform_width.hpp>
#include <boost/algorithm/string.hpp>

namespace MR
{

std::string encode64( const std::uint8_t * data, size_t size ) 
{
    MR_TIMER;
    using namespace boost::archive::iterators;
    using It = base64_from_binary<transform_width<const std::uint8_t *, 6, 8>>;
    auto tmp = std::string( It( data ), It( data + size ) );
    tmp.append( ( 3 - size % 3 ) % 3, '=' );
    return tmp;
}

std::vector<std::uint8_t> decode64( const std::string &val ) 
{
    MR_TIMER;
    using namespace boost::archive::iterators;
    using It = transform_width<binary_from_base64<std::string::const_iterator>, 8, 6>;
    #pragma warning(push)
        #pragma warning(disable:4127)
        return std::vector<std::uint8_t>( It( begin( val ) ), It( end( val ) ) );
    #pragma warning(pop)
}

} //namespace MR
