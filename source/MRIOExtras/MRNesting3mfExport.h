#pragma once
#include "config.h"
#ifndef MRIOEXTRAS_NO_XML
#include "exports.h"

#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRNestingStructures.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MRVector.h"

#include <filesystem>

namespace MR
{

namespace Nesting
{

struct Nesting3mfParams
{
    /// nested meshes with their transforms into nest
    Vector<MeshXf, ObjId> meshes;

    /// slicing step
    float zStep = 1.0f;

    /// if set: decimate slices
    bool decimateSlices{ true };

    /// optional image of all meshes in nest
    const Image* image = nullptr;

    /// optional screen shots of each mesh one by one
    const std::vector<Image>* meshImages{ nullptr };

    /// optional names of each mesh one by one
    const std::vector<std::string>* meshNames{ nullptr };

    ProgressCallback cb;
};

/// exports slicestack 3mf file based on placed meshes
MRIOEXTRAS_API Expected<void> exportNesting3mf( const std::filesystem::path& path, const Nesting3mfParams& params );

}

}
#endif
