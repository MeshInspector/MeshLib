#pragma once

#include "MRSpaceMouseParameters.h"
#include "MRMesh/MRMeshFwd.h"

namespace MR
{

// this class stores maps spacemouse event - program action
class SpaceMouseController
{
public:
    MR_ADD_CTOR_DELETE_MOVE( SpaceMouseController );
    void connect();

    MRVIEWER_API void setParameters( const SpaceMouseParameters& newParams );
    MRVIEWER_API SpaceMouseParameters getParameters() const;

private:
    bool spaceMouseMove_( const Vector3f& translate, const Vector3f& rotate );
    bool spaceMouseDown_( int key );

    bool lockRotate_{ false };
    bool showKeyDebug_{ false };

    SpaceMouseParameters params;
};

} //namespace MR
