#pragma once

#include "MRSpaceMouseParameters.h"
#include "MRMesh/MRMeshFwd.h"

namespace MR::SpaceMouse
{

// this class stores maps spacemouse event - program action
class Controller
{
public:
    MR_ADD_CTOR_DELETE_MOVE( Controller );
    void connect();

    void setParameters( const Parameters& newParams ) { params_ = newParams; }
    const Parameters& getParameters() const { return params_; }

private:
    bool spaceMouseMove_( const Vector3f& translate, const Vector3f& rotate );
    bool spaceMouseDown_( int key );

    bool lockRotate_{ false };
    bool showKeyDebug_{ false };

    Parameters params_;
};

} //namespace MR
