#pragma once

#include "MRMeshFwd.h"
#include "MRPch/MRBindingMacros.h"
#include <filesystem>
#include <cstdio>

namespace MR
{

/// this version of fopen unlike std::fopen supports unicode file names on Windows
MR_BIND_IGNORE MRMESH_API FILE * fopen( const std::filesystem::path & filename, const char * mode );

/// the class to open C FILE handle and automatically close it in the destructor
class MR_BIND_IGNORE File
{
public:
    File() = default;
    File( const File & ) = delete;
    File( File && r ) : handle_( r.handle_ ) { r.detach(); }
    File( const std::filesystem::path & filename, const char * mode ) { open( filename, mode ); }
    ~File() { close(); }

    File& operator =( const File & ) = delete;
    File& operator =( File && r ) { close(); handle_ = r.handle_; r.detach(); return * this; }

    operator FILE *() const { return handle_; }

    MRMESH_API FILE * open( const std::filesystem::path & filename, const char * mode );
    MRMESH_API void close();

    /// the user takes control over the handle
    void detach() { handle_ = nullptr; }
    /// gives control over the handle to this object
    void attach( FILE * h ) { if ( handle_ != h ) { close(); handle_ = h; } }

private:
    FILE * handle_ = nullptr;
};

} // namespace MR
