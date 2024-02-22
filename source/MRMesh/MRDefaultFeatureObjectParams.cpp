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

    if ( auto point = dynamic_cast<ObjectPointsHolder*>( &object ) )
        point->setPointSize( 10 );
    else if ( auto line = dynamic_cast<ObjectLinesHolder*>( &object ) )
        line->setLineWidth( 3 );
    else
        object.setGlobalAlpha( 128 );
}

}
