#pragma once
#include "exports.h"
#include "MRMesh\MRProgressCallback.h"

namespace MR
{
class VisualObject;

MRVIEWER_API std::string saveObjectToFile( const std::shared_ptr<VisualObject>& obj, const std::filesystem::path& filename,
                                           ProgressCallback callback = emptyProgressCallback );

#ifdef __EMSCRIPTEN__

extern "C" {

MRVIEWER_API EMSCRIPTEN_KEEPALIVE int load_files( int count, const char** filenames );

MRVIEWER_API EMSCRIPTEN_KEEPALIVE int save_file( const char* filename );

MRVIEWER_API EMSCRIPTEN_KEEPALIVE int save_scene( const char* filename );

}
#endif

}
