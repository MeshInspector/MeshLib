#pragma once

namespace MR
{

// the purpose of this struct is to invalidate object cache in its destructor
template<class T>
struct Writer
{
    T & obj;
    Writer( T & o ) : obj( o ) { }
    ~Writer() { obj.invalidateCaches(); }
};

#define MR_WRITER( obj ) MR::Writer _writer( obj );

} //namespace MR
