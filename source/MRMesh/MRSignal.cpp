#include "MRSignal.h"

namespace MR
{

template<typename T>
boost::signals2::connection Signal<T>::connect( const boost::function<T> & slot, boost::signals2::connect_position position )
{
    return Parent::connect( slot, position );
}

#define INSTANTIATE(T) \
    template boost::signals2::connection Signal<T>::connect( const boost::function<T> & slot, boost::signals2::connect_position position );

INSTANTIATE(void())
INSTANTIATE(void(uint32_t))

} //namespace MR
