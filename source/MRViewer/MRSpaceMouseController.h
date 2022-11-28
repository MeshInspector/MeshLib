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

	struct Params
	{
		Vector3f translateScale{ 50.f, 50.f, 50.f }; // range [1; 100]
		Vector3f rotateScale{ 50.f, 50.f, 50.f }; // range [1; 100]
	};

	MRVIEWER_API void setParams( const Params& newParams );
	MRVIEWER_API Params getParams() const;

private:
	bool spaceMouseMove_( const Vector3f& translate, const Vector3f& rotate );
	bool spaceMouseDown_( int key );

	bool lockRotate_{ false };
	bool showKeyDebug_{ false };

	Params params;
};

}
