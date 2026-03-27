#pragma once
#include "MRNestingStructures.h"
#include "MRMesh/MRVector.h"
#include "MRMesh/MRId.h"
#include "MRMesh/MRExpected.h"

namespace MR
{

namespace Nesting
{

struct BoxNestingCorner
{
    /// Vector3f - corner position
    Vector3f pos;
    /// corner mask 0bZYX (0b000 - lower left corner, 0b111 - upper right)
    uint8_t bitMask{ 0 };
};

/// class to override box nesting metrics
class MRMESH_CLASS IBoxNestingPriority
{
public:
    virtual ~IBoxNestingPriority() = default;

    /// init priority calculation with box of placed object 
    virtual void init( const Box3f& thisBox ) = 0;

    /// accumulate priority by one of already nested boxes
    virtual void addNested( const Box3f& box ) = 0;

    /// finalize priority and return the value (more - better)
    virtual double complete() const = 0;
};

/// priority metric that minimizes position of new object by Z->Y->X coordinate in nest
MRMESH_API std::shared_ptr<IBoxNestingPriority> getNestPostionMinPriorityMetric( const Box3f& nest );

/// priority metric that maximizes density of placement in local neighborhood
MRMESH_API std::shared_ptr<IBoxNestingPriority> getNeighborigDensityPriorityMetric( const Box3f& nest, float neighborhood );

struct BoxNestingOptions
{
    /// if true allows placing objects over the bottom plane
    bool allow3dNesting{ true };

    /// set false to keep original XY orientation
    bool allowRotation{ false };

    /// if true - nests objects in the order of decreasing volume, otherwise nest in the input order
    bool volumeBasedOrder = true;

    /// reduces nesting candidate options for speedup
    bool checkLessCombinations{ false };

    /// limit maximum number of tries, not to freeze for too long in this function
    int iterationLimit = 10'000'000;

    /// metric to calculate priority for candidates placement
    /// if not set - default `getNestPostionMinPriorityMetric` is used
    std::shared_ptr<IBoxNestingPriority> priorityMetric;

    /// optional input expansion of boxes (useful to compensate shrinkage of material)
    const Vector3f* expansionFactor{ nullptr };

    /// if not-nullptr contains boxes that are fixed in the nest and should not be intersected by floating (input) meshes
    const std::vector<Box3f>* preNestedVolumes{ nullptr };

    /// user might force these sockets to be considered as corners for placing candidates
    const std::vector<BoxNestingCorner>* additinalSocketCorners{ nullptr };

    /// callback indicating progress of the nesting
    ProgressCallback cb;
};

/// fills `outCorners` based on `nestedBoxes` corners \n
/// also adding corners in intersections of `nestedBoxes`
MRMESH_API Expected<void> fillNestingSocketCorneres( const std::vector<Box3f>& nestedBoxes, std::vector<BoxNestingCorner>& outCorners, const ProgressCallback& cb = {} );

struct BoxNestingParams
{
    NestingBaseParams baseParams;
    BoxNestingOptions options;
};

/// finds best positions of input meshes to fit the nest (checks them by contacting box corners)
MRMESH_API Expected<Vector<NestingResult, ObjId>> boxNesting( const Vector<MeshXf, ObjId>& meshes, const BoxNestingParams& params );

}

}