#pragma once

#include "exports.h"
#include "MRMesh/MRFlagOperators.h"
#include "MRMesh/MRICP.h"

namespace MR
{

// Helper functions to draw ICP properties dialog
namespace ICPDialog
{

// Type of parameters changed by user
enum class ChangeFlags
{
    NoChanges         = 0,
    ParamsChanging    = 1,  // Parameters have been changed but the user has not released the controls yet
    ParamsModified    = 2,  // The user has finished the modification
    PointPairsChanged = 4,  // Need to recalculate point pairs
    SamplingChanged   = 8   // Need to resample points
};
MR_MAKE_FLAG_OPERATORS( ChangeFlags )

// Recommended width of dialog (should be scaled with `UI::scale()`)
constexpr float defaultWidth = 320.0f;

enum class SampleTarget : int
{
    SampleBoth, ///< sample points on both floating and reference objects
    SampleFlt,  ///< sample points on floating object only
    SampleRef,  ///< sample points on reference object only
};

// Draw pairs generation settings;
// descriptorSpacing - if given, the spacing of the samples where Auto Initial computes shape descriptors (FPFH), its change sets no flags
MRVIEWER_API ChangeFlags drawPairsFormingSettings( ICPProperties& params, float& samplingVoxelSize,
    float objectSize, bool *expanded = nullptr, SampleTarget * sampleTarget = nullptr, bool rigidModeOnly = false, bool drawCollapsingHeader = true,
    bool * registerWholeObjects = nullptr, float * descriptorSpacing = nullptr );

// Default descriptor spacing: a third of the default sampling spacing, to capture small details like teeth
[[nodiscard]] MRVIEWER_API float defaultDescriptorSpacing( float objectSize );

// Draw stop condition section
MRVIEWER_API ChangeFlags drawStopConditions( ICPProperties& params, bool* expanded = nullptr, bool drawCollapsingHeader = true );

// Draw point pairs information section
MRVIEWER_API void drawInformation( size_t numSamples, size_t numPairs, float meanSqDistToPoint, float meanSqDistToPlane, bool* expanded = nullptr );

// Draw ICP result section
MRVIEWER_API void drawResult( std::string resultString );

}
}
