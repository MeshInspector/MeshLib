#include "MRSignal.h"

namespace MR
{

template<typename Signature, typename Combiner>
boost::signals2::connection Signal<Signature, Combiner>::connect( const boost::function<Signature> & slot, boost::signals2::connect_position position )
{
    return Parent::connect( slot, position );
}

#define INSTANTIATE(Signature) \
    template boost::signals2::connection Signal<Signature>::connect( const boost::function<Signature> & slot, boost::signals2::connect_position position );

INSTANTIATE(void())
INSTANTIATE(void(uint32_t))

#define INSTANTIATE2(Signature, Combiner) \
    template boost::signals2::connection Signal<Signature, Combiner>::connect( const boost::function<Signature> & slot, boost::signals2::connect_position position );

} //namespace MR
