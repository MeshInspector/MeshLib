#pragma once

#include "MRExpected.h"
#include "MRMeshFwd.h"

namespace Json { class Value; }

namespace MR
{

struct EndMillCutter
{
    enum class Type
    {
        Flat,
        Ball,
        // TODO: bull nose
        // TODO: chamfer
        Count
    };

    Type type;
    float radius;

    MRMESH_API static EndMillCutter makeFlat( float radius );
    MRMESH_API static EndMillCutter makeBall( float radius );

    MRMESH_API static Expected<EndMillCutter> deserialize( const Json::Value& root );
    MRMESH_API void serialize( Json::Value& root ) const;
};

struct EndMillTool
{
    float length;
    EndMillCutter cutter;

    /// ...
    [[nodiscard]] MRMESH_API Mesh toMesh( float minEdgeLen = 0.f ) const;

    MRMESH_API static Expected<EndMillTool> deserialize( const Json::Value& root );
    MRMESH_API void serialize( Json::Value& root ) const;
};

} // namespace MR
