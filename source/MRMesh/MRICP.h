#pragma once
#include "MRMeshFwd.h"
#include "MRAligningTransform.h"
#include "MRVector3.h"
#include "MRMesh.h"
#include "MRId.h"

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
};

struct ICPProperties
{
    ICPMethod method = ICPMethod::PointToPlane;
    // rotation part will be limited by this value. If the whole rotation exceed this value, it will be normalized to that.
    // Note: PointToPlane only!
    float p2plAngleLimit = PI_F / 6.0f;
    // Points pair will be counted only if cosine between surface normals in points is higher
    float cosTreshold = 0.7f;
    // Points pair will be counted only if squared distance between points is lower
    float distTresholdSq = 1.f;
    // Sigma multiplier for statistic throw of paints pair based on the distance
    // Default: all pairs in the interval the (distance = mean +- 3*sigma) are passed
    float distStatisticSigmaFactor = 3.f;
    // Finds only translation. Rotation part is identity matrix
    ICPMode icpMode = ICPMode::AnyRigidXf;
    // If this vector is not zero then rotation is allowed relative to this axis only
    Vector3f fixedRotationAxis;
    // keep point pairs from first iteration
    bool freezePairs = false;

    // parameters of iterative call
    int iterLimit = 10; // maximum iterations
    int badIterStopCount = 3; // maximum iterations without improvements

    // Algorithm target minimization criteria
    // Mean distance between points for p2pt, mean distance to planes for p2pl
    float exitVal = 0.;
};

// This class allows to match two meshes with almost same geometry throw ICP point-to-point or point-to-plane algorithms
class MeshICP
{
public:
    // xf parameters should represent current transformations of meshes
    // initMeshXf transform from the local mesh basis to the global. 
    // refMeshXf transform from the local refMesh basis to the global
    // calculateTransform returns new mesh transformation to the global frame, which matches refMesh in the global frame
    // bitset allows to take exact set of vertices from the mesh
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

    const ICPProperties& getParams() const { return prop_; }
    MRMESH_API Vector3f getShiftVector() const; // shows mean pair vector
    MRMESH_API std::string getLastICPInfo() const; // returns status info string
    MRMESH_API float getMeanSqDistToPoint() const; // shows current mean square deviation
    MRMESH_API float getMeanSqDistToPlane() const; //shows current P2Pl metrics
    const std::vector<VertPair>& getVertPairs() const { return vertPairs_; } // used to visualize generated points pairs
    MRMESH_API std::pair<float, float> getDistLimitsSq() const; // finds squared minimum and maximum pairs distances

    //returns new xf transformation for the floating mesh, which allows to match reference mesh
    MRMESH_API AffineXf3f calculateTransformation();
    MRMESH_API void updateVertPairs();

private:
    // input meshes variables
    MeshPart meshPart_;
    AffineXf3f xf_;
    
    VertBitSet bitSet_; // region of interests on the floating mesh
    
    MeshPart refPart_;
    AffineXf3f refXf_;
    AffineXf3f refXfInv_; // optimized for reference points transformation

    ICPProperties prop_;
    std::unique_ptr<PointToPointAligningTransform> p2pt_ = std::make_unique<PointToPointAligningTransform>();
    std::unique_ptr<PointToPlaneAligningTransform> p2pl_ = std::make_unique<PointToPlaneAligningTransform>();

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
