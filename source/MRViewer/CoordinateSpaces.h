#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector3.h"
#include "imgui/imgui.h"

#define DEFAULT_CTORS( x ) x() = default

namespace MR
{

struct WorldPoint
{
	DEFAULT_CTORS( WorldPoint );
	WorldPoint( const Vector3f& point );
	explicit WorldPoint( const CameraPoint& cp );

	Vector3f p;
};

struct CameraPoint
{
	DEFAULT_CTORS( CameraPoint );
	explicit CameraPoint( const WorldPoint& wp );
	explicit CameraPoint( const ClipPoint& wp );

	Vector3f p;
}

struct ClipPoint
{
	DEFAULT_CTORS( CameraPoint );
	explicit ClipPoint( const CameraPoint& wp );
	explicit ClipPoint( const ViewportPoint& wp, const Viewport& vp );

	Vector3f p;
};

struct ViewportPoint
{
	DEFAULT_CTORS( ViewportPoint );
	explicit ViewportPoint( const ClipPoint& wp, const Viewport& vp );
	explicit ViewportPoint( const FrameBufferPoint& wp );

	Vector3f p;
};

struct FrameBufferPoint
{
	DEFAULT_CTORS( FrameBufferPoint );
	explicit FrameBufferPoint( const ViewportPoint& wp );
	explicit FrameBufferPoint( const WindowPoint& wp );
	explicit FrameBufferPoint( const ImGuiPoint& wp );

	Vector2f p;
};

struct WindowPoint
{
	DEFAULT_CTORS( WindowPoint );
	explicit WindowPoint( const FrameBufferPoint& wp );

	Vector2f p;
};

struct ImGuiPoint
{
	DEFAULT_CTORS( ImGuiPoint );
	explicit ImGuiPoint( const FrameBufferPoint& wp );

	operator ImVec2();

	Vector2f p;
};

}
