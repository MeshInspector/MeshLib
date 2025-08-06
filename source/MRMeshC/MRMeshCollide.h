#pragma once
#include "MRMeshFwd.h"
#include "MRMeshPart.h"

MR_EXTERN_C_BEGIN

/**
 * \brief checks that arbitrary mesh part A is inside of closed mesh part B
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 */
MRMESHC_API bool mrIsInside( const MRMeshPart* a, const MRMeshPart* b, const MRAffineXf3f* rigidB2A );

MR_EXTERN_C_END
