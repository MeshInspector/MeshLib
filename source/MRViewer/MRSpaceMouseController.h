#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "MRViewerFwd.h"
#include "MRMesh/MRVector3.h"

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

	Vector3f translateScale{ 1.f, 1.f, 1.f };
	Vector3f rotateScale{ 1.f, 1.f, 1.f };
};

}
