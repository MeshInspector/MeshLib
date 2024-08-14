#pragma once
#include "MRMeshFwd.h"
#include "MRICP.h"

MR_DOTNET_NAMESPACE_BEGIN

/// Parameters that are used for sampling of the MultiwayICP objects
public ref struct MultiwayICPSamplingParameters
{
    /// sampling size of each object
    float samplingVoxelSize = 0.0f;

    /// size of maximum icp group to work with
    /// if number of objects exceeds this value, icp is applied in cascade mode
    int maxGroupSize = 64;

    enum class CascadeMode
    {
        Sequential, /// separates objects on groups based on their index in ICPObjects (good if all objects about the size of all objects together)
        AABBTreeBased /// builds AABB tree based on each object bounding box and separates subtrees (good if each object much smaller then all objects together)
    } cascadeMode{ CascadeMode::AABBTreeBased };
};

/// Stores a pair of points: one samples on the source and the closest to it on the target
public ref class MultiwayICP
{
public:
    MultiwayICP( List<MeshOrPointsXf>^ objs, MultiwayICPSamplingParameters^ samplingParams );
    ~MultiwayICP();

    List<AffineXf3f^>^ CalculateTransformations();
    void ResamplePoints( MultiwayICPSamplingParameters^ samplingParams );
    bool UpdateAllPointPairs();
    void SetParams( ICPProperties^ props );
    float GetMeanSqDistToPoint();
    float GetMeanSqDistToPoint( double value );
    /// computes root-mean-square deviation from points to target planes
    float GetMeanSqDistToPlane();
    float GetMeanSqDistToPlane( double value );
    /// returns current pairs formed from samples on floating object and projections on reference object
     /// computes the number of samples able to form pairs
    int GetNumSamples();
    /// computes the number of active point pairs
    int GetNumActivePairs();

private:
    MR::MultiwayICP* icp_;
};

MR_DOTNET_NAMESPACE_END

