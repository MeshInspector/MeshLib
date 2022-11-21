#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "MRViewerFwd.h"

namespace MR
{

// this class stores maps spacemouse event - program action
class SpaceMouseController
{
public:
	MR_ADD_CTOR_DELETE_MOVE( SpaceMouseController );
	void connect();

private:
	bool spaceMouseMove_( const Vector3f& translate, const Vector3f& rotate );
	bool spaceMouseDown_( int key );

	bool lockRotate_{ false };
	bool showKeyDebug_{ false };
};

}
