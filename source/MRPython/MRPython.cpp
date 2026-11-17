#include "MRPython.h"
#include "MRUnifiedPythonStream.h"

#include <iostream>

template<StreamType T>
struct NumWritten
{
    static inline int counter = 0;
};

template<StreamType T>
void PythonStreamRedirector<T>::write( const std::string& text )
{
    ++NumWritten<T>::counter;
    MR::UnifiedPythonStream::get() << text;
    if constexpr ( T == Stdout )
        std::cout << text;
    else
        std::cerr << text;
}

template<StreamType T>
void PythonStreamRedirector<T>::flush()
{
    if constexpr ( T == Stdout )
        std::cout.flush();
    else
        std::cerr.flush();
}

template<StreamType T>
int PythonStreamRedirector<T>::getNumWritten()
{
    return NumWritten<T>::counter;
}

template class PythonStreamRedirector<Stdout>;
template class PythonStreamRedirector<Stderr>;
