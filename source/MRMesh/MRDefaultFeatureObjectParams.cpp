#include "MRDefaultFeatureObjectParams.h"

#include "MRMesh/MRObjectLinesHolder.h"
#include "MRMesh/MRObjectPointsHolder.h"
#include "MRMesh/MRVisualObject.h"

namespace MR
{

void setDefaultFeatureObjectParams( VisualObject& object )
{
    object.setFrontColor( Color( 216, 128, 70, 255 ), false );
    object.setFrontColor( Color( 193, 40, 107, 255 ), true );
    object.setBackColor( Color( 176, 124, 91, 255 ) );

    // This one is a bit weird. Feature rendering overrides alpha anyway,
    // but we need the object itself to have non-100% alpha for the rendering to be done correctly.
    object.setGlobalAlpha( 128 );
}

}
