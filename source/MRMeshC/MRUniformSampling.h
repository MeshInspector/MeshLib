#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

typedef struct MRUniformSamplingSettings
{
    /// minimal distance between samples
    float distance;
    /// if point cloud has normals then automatically decreases local distance to make sure that all points inside have absolute normal dot product not less than this value;
    /// this is to make sampling denser in the regions of high curvature;
    /// value <=0 means ignore normals;
    /// value >=1 means select all points (practically useless)
    float minNormalDot;
    /// if true process the points in lexicographical order, which gives tighter and more uniform samples;
    /// if false process the points according to their ids, which is faster
    bool lexicographicalOrder;
    // TODO: pNormals
    /// to report progress and cancel processing
    MRProgressCallback progress;
} MRUniformSamplingSettings;

MRMESHC_API MRUniformSamplingSettings mrUniformSamplingSettingsNew( void );

MRMESHC_API MRVertBitSet* mrPointUniformSampling( const MRPointCloud* pointCloud, const MRUniformSamplingSettings* settings );

MR_EXTERN_C_END
