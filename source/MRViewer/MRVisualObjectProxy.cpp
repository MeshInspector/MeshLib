#include "MRVisualObjectProxy.h"

#include "MRVisualObjectTag.h"
#include "MRMesh/MRColor.h"
#include "MRMesh/MRVisualObject.h"

namespace MR
{

Color VisualObjectProxy::getFrontColor( const VisualObject& visObj, bool selected, ViewportId viewportId )
{
    const auto& storage = VisualObjectTagManager::instance().storage();
    for ( const auto& data : visObj.getMetadata() )
    {
        constexpr std::string_view cTagPrefix { "tag=" };
        if ( data.starts_with( cTagPrefix ) )
        {
            const auto name = data.substr( cTagPrefix.size() );
            if ( auto it = storage.find( name ); it != storage.end() )
            {
                const auto& [_, tag] = *it;
                return ( selected ? tag.selectedColor : tag.unselectedColor ).get( viewportId );
            }
        }
    }

    return visObj.getFrontColor( selected, viewportId );
}

} // namespace MR
