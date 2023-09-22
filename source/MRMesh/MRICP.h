#pragma once

#include "MRMeshPart.h"
#include "MRMatrix3.h"
#include "MRId.h"
#include "MRConstants.h"
#include "MRAffineXf.h"
#include "MRBitSet.h"

namespace MR
{

enum class ICPMethod
{
    Combined = 0, // PointToPoint for the first 2 iterations, PointToPlane then
    PointToPoint = 1, // use it in the cases with big differences, takes more iterations
    PointToPlane = 2 // finds solution faster in fewer iterations
};

// You could fix any axis(axes) of rotation by using this modes
enum class ICPMode
{
    AnyRigidXf, // all 6 degrees of freedom (dof)
    OrthogonalAxis, // 5 dof, except argument axis
    FixedAxis, // 4 dof, translation and one argument axis
    TranslationOnly // 3 dof, no rotation
};

struct VertPair
{
    // coordinates of the closest point on reference mesh (after applying refXf)
    Vector3f refPoint;
    // surface normal in a vertex on the floating mesh (after applying Xf)
    Vector3f norm;
    // surface normal in a vertex on the reference mesh (after applying Xf)
    Vector3f normRef;
    // ID of the floating mesh vertex (usually applying Xf required)
    VertId vertId;
    // This is cosine between normals in first(floating mesh) and second(reference mesh) points
    // It evaluates how good is this pair
    float normalsAngleCos = 1.f;
    // Storing squared distance between vertices
    float vertDist2 = 0.f;
    // weight of the pair with respect to the sum of adjoining triangles square
    float weight = 1.f;

    friend bool operator == ( const VertPair&, const VertPair& ) = default;
};

struct ICPProperties
{
    ICPMethod method = ICPMethod::PointToPlane;
    // rotation part will be limited by this value. If the whole rotation exceed this value, it will be normalized to that.
    // Note: PointToPlane only!
    float p2plAngleLimit = PI_F / 6.0f; // [radians]
    // Points pair will be counted only if cosine between surface normals in points is higher
    float cosTreshold = 0.7f; // in [-1,1]
    // Points pair will be counted only if squared distance between points is lower than
    float distTresholdSq = 1.f; // [distance^2]
    // Sigma multiplier for statistic throw of paints pair based on the distance
    // Default: all pairs in the interval the (distance = mean +- 3*sigma) are passed
    float distStatisticSigmaFactor = 3.f; // dimensionless
    // Finds only translation. Rotation part is identity matrix
    ICPMode icpMode = ICPMode::AnyRigidXf;
    // If this vector is not zero then rotation is allowed relative to this axis only
    Vector3f fixedRotationAxis;
    // keep point pairs from first iteration
    bool freezePairs = false;

    // parameters of iterative call
    int iterLimit = 10; // maximum iterations
    int badIterStopCount = 3; // maximum iterations without improvements

    // Algorithm target root-mean-square distance. As soon as it is reached, the algorithm stops.
    float exitVal = 0; // [distance]
};

// This class allows to match two meshes with almost same geometry using ICP point-to-point or point-to-plane algorithms
class MeshICP
{
public:
    // xf parameters should represent current transformations of meshes
    // fltMeshXf transform from the local floatingMesh basis to the global
    // refMeshXf transform from the local referenceMesh basis to the global
    // floatingMeshBitSet allows to take exact set of vertices from the mesh
    MRMESH_API MeshICP(const MeshPart& floatingMesh, const MeshPart& referenceMesh, const AffineXf3f& fltMeshXf, const AffineXf3f& refMeshXf,
        const VertBitSet& floatingMeshBitSet);
    MRMESH_API MeshICP(const MeshPart& floatingMesh, const MeshPart& referenceMesh, const AffineXf3f& fltMeshXf, const AffineXf3f& refMeshXf,
        float floatSamplingVoxelSize ); // positive value here defines voxel size, and only one vertex per voxel will be selected
    // TODO: add single transform constructor

    // tune algirithm params before run calculateTransformation()
    void setParams(const ICPProperties& prop) { prop_ = prop; }
    MRMESH_API void setCosineLimit(const float cos);
    MRMESH_API void setDistanceLimit( const float dist );
    MRMESH_API void setBadIterCount( const int iter );
    MRMESH_API void setPairsWeight(const std::vector<float> w);
    MRMESH_API void setDistanceFilterSigmaFactor(const float factor);
    MRMESH_API void recomputeBitSet(const float floatSamplingVoxelSize);
    MRMESH_API void setXfs( const AffineXf3f& fltMeshXf, const AffineXf3f& refMeshXf );
    MRMESH_API void setFloatXf( const AffineXf3f& fltMeshXf );
    // recompute point pairs after manual change of transformations or parameters
    MRMESH_API void updateVertPairs();

    const ICPProperties& getParams() const { return prop_; }
    MRMESH_API Vector3f getShiftVector() const; // shows mean pair vector
    MRMESH_API std::string getLastICPInfo() const; // returns status info string
    MRMESH_API float getMeanSqDistToPoint() const; // computes root-mean-square deviation between points
    MRMESH_API float getMeanSqDistToPlane() const; // computes root-mean-square deviation from points to target planes
    const std::vector<VertPair>& getVertPairs() const { return vertPairs_; } // used to visualize generated points pairs
    MRMESH_API std::pair<float, float> getDistLimitsSq() const; // finds squared minimum and maximum pairs distances

    // returns new xf transformation for the floating mesh, which allows to match reference mesh
    MRMESH_API AffineXf3f calculateTransformation();

private:
    MeshPart floatMesh_;
    AffineXf3f floatXf_;
    VertBitSet floatVerts_; ///< vertices of floating object to find their pairs on reference mesh
    
    MeshPart refMesh_;
    AffineXf3f refXf_;

    AffineXf3f float2refXf_; ///< transformation from floating object space to reference object space

    ICPProperties prop_;

    std::vector<VertPair> vertPairs_;

    // types of exit conditions in calculation
    enum class ExitType {
        NotStarted, // calculation is not started yet
        NotFoundSolution, // solution not found in some iteration
        MaxIterations, // iteration limit reached
        MaxBadIterations, // limit of non-improvement iterations in a row reached
        StopMsdReached // stop mean square deviation reached
    };
    ExitType resultType_{ ExitType::NotStarted };

    void removeInvalidVertPairs_();

    void updateVertFilters_();

    int iter_ = 0;
    bool p2ptIter_();
    bool p2plIter_();
};

}
