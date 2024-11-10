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
MRMESH_API std::streamoff getStreamSize( std::istream& in );

// reads input stream to string
MRMESH_API Expected<std::string> readString( std::istream& in );

// reads input stream to monolith char block
MR_BIND_IGNORE MRMESH_API Expected<Buffer<char>> readCharBuffer( std::istream& in );

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
