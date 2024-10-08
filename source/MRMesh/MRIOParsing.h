#pragma once
#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRBuffer.h"
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
MRMESH_API Expected<Buffer<char>> readCharBuffer( std::istream& in );

// read coordinates to `v` separated by space
template<typename T>
VoidOrErrStr parseTextCoordinate( const std::string_view& str, Vector3<T>& v, Vector3<T>* n = nullptr, Color* c = nullptr );
template<typename T>
VoidOrErrStr parseObjCoordinate( const std::string_view& str, Vector3<T>& v, Vector3<T>* c = nullptr );
template<typename T>
VoidOrErrStr parsePtsCoordinate( const std::string_view& str, Vector3<T>& v, Color& c );

// reads the first integer number in the line
MRMESH_API VoidOrErrStr parseFirstNum( const std::string_view& str, int& num );
// reads the polygon points and optional number of polygon points
// example
// N vertex0 vertex1 ... vertexN
MRMESH_API VoidOrErrStr parsePolygon( const std::string_view& str, VertId* vertId, int* numPoints );

template<typename T>
[[deprecated( "use parseTextCoordinate() instead")]]
VoidOrErrStr parseAscCoordinate( const std::string_view& str, Vector3<T>& v, Vector3<T>* n = nullptr, Color* c = nullptr );



template<typename T>
VoidOrErrStr parseSingleNumber( const std::string_view& str, T& num );

}