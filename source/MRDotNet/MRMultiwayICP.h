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

/// This class allows you to register many objects having similar parts
/// and known initial approximations of orientations/locations using
/// Iterative Closest Points (ICP) point-to-point or point-to-plane algorithms
public ref class MultiwayICP
{
public:
    MultiwayICP( List<MeshOrPointsXf>^ objs, MultiwayICPSamplingParameters^ samplingParams );
    ~MultiwayICP();

    /// runs ICP algorithm given input objects, transformations, and parameters;
    /// \return adjusted transformations of all objects to reach registered state
    List<AffineXf3f^>^ CalculateTransformations();
    /// select pairs with origin samples on all objects
    void ResamplePoints( MultiwayICPSamplingParameters^ samplingParams );
    /// in each pair updates the target data and performs basic filtering (activation)
    /// in cascade mode only useful for stats update
    bool UpdateAllPointPairs();
    /// tune algorithm params before run calculateTransformations()
    void SetParams( ICPProperties^ props );
    /// computes root-mean-square deviation between points
    float GetMeanSqDistToPoint();
    /// computes the standard deviation from given value
    float GetMeanSqDistToPoint( double value );
    /// computes root-mean-square deviation from points to target planes
    float GetMeanSqDistToPlane();
    /// computes the standard deviation from given value
    float GetMeanSqDistToPlane( double value );
    /// computes the number of samples able to form pairs
    int GetNumSamples();
    /// computes the number of active point pairs
    int GetNumActivePairs();

private:
    MR::MultiwayICP* icp_;
};

MR_DOTNET_NAMESPACE_END

