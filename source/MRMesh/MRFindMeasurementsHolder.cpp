#include "MRFindMeasurementsHolder.h"

namespace MR
{

static constexpr const char* objectName = "Measurements";

std::shared_ptr<Object> findOrCreateMeasurementsHolder( Object& object )
{
    auto ret = findMeasurementsHolderOpt( object );
    if ( !ret )
    {
        ret = std::make_shared<Object>();
        ret->setName( objectName );
        object.addChild( ret );
    }
    return ret;
}

std::shared_ptr<Object> findMeasurementsHolderOpt( Object& object )
{
    return object.find( objectName );
}

}
