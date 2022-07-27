#pragma once
#include "MRMesh/MRProgressCallback.h"
#include "MRMeshFwd.h"
#include <ostream>

namespace MR
{

/**
 * \brief write dataSize bytes from data to out stream by blocks blockSize bytes
 * \details if progress callback is not set, write all data by one block
 * \return false if process was canceled (callback is set and return false )
 */
MRMESH_API bool writeByBlocks( std::ostream& out, const char* data, size_t dataSize, ProgressCallback callback = {}, size_t blockSize = ( size_t( 1 ) << 16 ) );

}
