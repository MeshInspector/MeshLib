#pragma once

#include "MRExpected.h"
#include "MRMeshFwd.h"

namespace Json { class Value; }

namespace MR
{

/// end mill cutter specifications
struct EndMillCutter
{
    /// cutter type
    enum class Type
    {
        /// flat end mill
        Flat,
        /// ball end mill
        Ball,
        /// bull nose end mill
        BullNose,
        /// chamfer end mill
        Chamfer,

        Count
    };
    Type type = Type::Flat;
    /// (bull nose) corner radius
    float cornerRadius = 0.f;
    /// (chamfer) cutting angle
    float cuttingAngle = 0.f;
    /// (chamfer) end diameter
    float endDiameter = 0.f;
};

/// end mill tool specifications
struct EndMillTool
{
    /// overall tool length
    float length = 1.f;
    /// tool diameter
    float diameter = 0.1f;
    /// cutter
    EndMillCutter cutter;

    /// compute the minimal cut length based on the cutter parameters
    [[nodiscard]] MRMESH_API float getMinimalCutLength() const;

    /// create a tool mesh
    [[nodiscard]] MRMESH_API Mesh toMesh( int horizontalResolution = 32, int verticalResolution = 32 ) const;
};

MRMESH_API void serializeToJson( const EndMillCutter& cutter, Json::Value& root );
MRMESH_API void serializeToJson( const EndMillTool& tool, Json::Value& root );

MRMESH_API Expected<void> deserializeFromJson( const Json::Value& root, EndMillCutter& cutter );
MRMESH_API Expected<void> deserializeFromJson( const Json::Value& root, EndMillTool& tool );

} // namespace MR
