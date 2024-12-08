#pragma once

#include "config.h"
#ifndef MRIOEXTRAS_NO_3MF
#include "exports.h"

#include <MRMesh/MRExpected.h>
#include <MRMesh/MRObject.h>
#include <MRMesh/MRLoadedObjects.h>

namespace MR
{

// loads scene from 3MF file in a new container object
MRIOEXTRAS_API Expected<LoadedObject> deserializeObjectTreeFrom3mf( const std::filesystem::path& file, const ProgressCallback& callback = {} );

// loads scene from .model file in a new container object
MRIOEXTRAS_API Expected<LoadedObject> deserializeObjectTreeFromModel( const std::filesystem::path& file, const ProgressCallback& callback = {} );

} // namespace MR
#endif
