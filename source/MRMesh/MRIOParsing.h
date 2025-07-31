#pragma once
#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRBuffer.h"
#include "MRPch/MRBindingMacros.h"
#include <istream>

namespace MR
{

// returns offsets for each new line in monolith char block
MRMESH_API std::vector<size_t> splitByLines( const char* data, size_t size );

// get the size of the remaining data in the input stream
// This has `MR_BIND_IGNORE` for now because it doesn't look very useful in C, since our `istream` bindings are minimal (the only istream you
//   can access is `std::cin`). If we expand them later, this can be added back.
// Note that if you decide to un-ignore this, the return type has to be changed to our `Int64` typedef to avoid issues on Mac.
MRMESH_API MR_BIND_IGNORE std::streamoff getStreamSize( std::istream& in );

// reads input stream to string
// This has `MR_BIND_IGNORE` for now because it doesn't look very useful in C, since our `istream` bindings are minimal (the only istream you
//   can access is `std::cin`). If we expand them later, this can be added back.
MRMESH_API MR_BIND_IGNORE Expected<std::string> readString( std::istream& in );

// reads input stream to monolith char block
MRMESH_API MR_BIND_IGNORE Expected<Buffer<char>> readCharBuffer( std::istream& in );

// read coordinates to `v` separated by space
template<typename T>
Expected<void> parseTextCoordinate( const std::string_view& str, Vector3<T>& v, Vector3<T>* n = nullptr, Color* c = nullptr );
template<typename T>
Expected<void> parseObjCoordinate( const std::string_view& str, Vector3<T>& v, Vector3<T>* c = nullptr );
template<typename T>
Expected<void> parsePtsCoordinate( const std::string_view& str, Vector3<T>& v, Color& c );

// reads the first integer number in the line
MRMESH_API Expected<void> parseFirstNum( const std::string_view& str, int& num );
// reads the polygon points and optional number of polygon points
// example
// N vertex0 vertex1 ... vertexN
MRMESH_API Expected<void> parsePolygon( const std::string_view& str, VertId* vertId, int* numPoints );

template<typename T>
[[deprecated( "use parseTextCoordinate() instead")]]
Expected<void> parseAscCoordinate( const std::string_view& str, Vector3<T>& v, Vector3<T>* n = nullptr, Color* c = nullptr );



template<typename T>
Expected<void> parseSingleNumber( const std::string_view& str, T& num );

}
