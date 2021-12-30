#pragma once
#include "MRMeshFwd.h"

namespace MR
{
// Z is symmetry axis of this torus
// points - optional out points of main circle
MRMESH_API Mesh makeTorus( float primaryRadius = 1.0f, float secondaryRadius = 0.1f, int primaryResolution = 16, int secondaryResolution = 16,
						   std::vector<Vector3f>* points = nullptr );

// creates torus without inner half faces
// main application - testing fillHole and Stitch
MRMESH_API Mesh makeOuterHalfTorus( float primaryRadius = 1.0f, float secondaryRadius = 0.1f, int primaryResolution = 16, int secondaryResolution = 16,
						   std::vector<Vector3f>* points = nullptr );

// creates torus with inner protruding half as undercut
// main application - testing fixUndercuts
MRMESH_API Mesh makeTorusWithUndercut( float primaryRadius = 1.0f, float secondaryRadiusInner = 0.1f, float secondaryRadiusOuter = 0.2f, int primaryResolution = 16, int secondaryResolution = 16,
						   std::vector<Vector3f>* points = nullptr );

// creates torus with some handed-up points
// main application - testing fixSpikes and Relax
MRMESH_API Mesh makeTorusWithSpikes( float primaryRadius = 1.0f, float secondaryRadiusInner = 0.1f, float secondaryRadiusOuter = 0.5f, int primaryResolution = 16, int secondaryResolution = 16,
						   std::vector<Vector3f>* points = nullptr );

// creates torus with empty sectors
// main application - testing Components
MRMESH_API Mesh makeTorusWithComponents( float primaryRadius = 1.0f, float secondaryRadius = 0.1f, int primaryResolution = 16, int secondaryResolution = 16,
						   std::vector<Vector3f>* points = nullptr );

// creates torus with empty sectors
// main application - testing Components
MRMESH_API Mesh makeTorusWithSelfIntersections( float primaryRadius = 1.0f, float secondaryRadius = 0.1f, int primaryResolution = 16, int secondaryResolution = 16,
						   std::vector<Vector3f>* points = nullptr );
}