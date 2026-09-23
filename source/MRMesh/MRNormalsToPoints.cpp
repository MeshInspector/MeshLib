#include "MRNormalsToPoints.h"
#include "MRMesh.h"
#include "MRParallelFor.h"
#include "MRTimer.h"
#include "MRRelaxParams.h" //getLimitedPos
#include <limits>

#include <MRPch/MREigenSparseCore.h>
#include <Eigen/SparseCholesky>

namespace MR
{

namespace
{

class Solver : public NormalsToPoints::ISolver
{
public:
    virtual void prepare( const MeshTopology & topology, float guideWeight ) override;
    virtual void run( const VertCoords & guide, const FaceNormals & normals, VertCoords & points, float maxInitialDistSq ) override;

private:
    // fills mat_ with 3 guide equations per vertex and 2 equations per triangle: dot(p1-p0,n)=0, dot(p2-p0,n)=0
    void fillMatrix_( const FaceNormals * normals );

    const MeshTopology * topology_ = nullptr;
    float guideWeight_ = 0;
    int nVerts_ = 0;
    using SparseMatrix = Eigen::SparseMatrix<double,Eigen::RowMajor>;
    SparseMatrix mat_;
    Eigen::VectorXd rhs_;
    using SparseMatrixColMajor = Eigen::SparseMatrix<double,Eigen::ColMajor>;
    Eigen::SimplicialLDLT<SparseMatrixColMajor> ldlt_;
};

void Solver::fillMatrix_( const FaceNormals * normals )
{
    MR_TIMER;
    std::vector< Eigen::Triplet<double> > mTriplets;
    mTriplets.reserve( 3 * nVerts_ + 18 * topology_->numValidFaces() );
    // every point shall be close to corresponding guide point (with small weight)
    for ( int i = 0; i < 3 * nVerts_; ++i )
        mTriplets.emplace_back( i, i, guideWeight_ );

    // add 2 equations per triangle: its edges shall be orthogonal to the target normal
    const int nRows = 3 * nVerts_ + 2 * topology_->numValidFaces();
    int row = 3 * nVerts_;
    for ( auto f : topology_->getValidFaces() )
    {
        VertId vs[3];
        topology_->getTriVerts( f, vs );
        // without normals, only the sparsity pattern is important
        const auto n = normals ? Vector3d( (*normals)[f] ) : Vector3d::diagonal( 1 );
        for ( int j = 1; j < 3; ++j )
        {
            for ( int i = 0; i < 3; ++i )
            {
                mTriplets.emplace_back( row, 3 * vs[j] + i,  n[i] );
                mTriplets.emplace_back( row, 3 * vs[0] + i, -n[i] );
            }
            ++row;
        }
    }
    assert( row == nRows );

    mat_.resize( nRows, 3 * nVerts_ );
    mat_.setFromTriplets( mTriplets.begin(), mTriplets.end() );
}

void Solver::prepare( const MeshTopology & topology, float guideWeight )
{
    MR_TIMER;
    topology_ = &topology;
    guideWeight_ = guideWeight;
    nVerts_ = (int)topology.vertSize();

    fillMatrix_( nullptr );
    SparseMatrix A = mat_.adjoint() * mat_;
    ldlt_.analyzePattern( A );

    rhs_ = Eigen::VectorXd::Zero( mat_.rows() );
}

void Solver::run( const VertCoords & guide, const FaceNormals & normals, VertCoords & points, float maxInitialDistSq )
{
    MR_TIMER;
    assert( topology_ );
    if ( !topology_ )
        return;

    fillMatrix_( &normals );
    SparseMatrix A = mat_.adjoint() * mat_;
    ldlt_.factorize( A );

    // every point shall be close to corresponding guide point (with small weight),
    // the equations per triangle have zero right-hand side
    ParallelFor( 0_v, guide.endId(), [&]( VertId v )
    {
        for ( int i = 0; i < 3; ++i )
            rhs_[3 * v + i] = guideWeight_ * guide[v][i];
    } );

    // solve linear equations
    const Eigen::VectorXd sol = ldlt_.solve( mat_.adjoint() * rhs_ );

    // copy back the solution into points
    const bool limitNearInitial = std::isfinite( maxInitialDistSq );
    ParallelFor( 0_v, guide.endId(), [&]( VertId v )
    {
        Vector3f np;
        for ( int i = 0; i < 3; ++i )
            np[i] = (float)sol[3 * v + i];
        if ( limitNearInitial )
            np = getLimitedPos( np, guide[v], maxInitialDistSq );
        points[v] = np;
    } );
}

} //anonymous namespace

void NormalsToPoints::prepare( const MeshTopology & topology, float guideWeight )
{
    solver_ = std::make_unique<Solver>();
    solver_->prepare( topology, guideWeight );
}

void NormalsToPoints::run( const VertCoords & guide, const FaceNormals & normals, VertCoords & points )
{
    run( guide, normals, points, std::numeric_limits<float>::infinity() );
}

void NormalsToPoints::run( const VertCoords & guide, const FaceNormals & normals, VertCoords & points, float maxInitialDistSq )
{
    assert( solver_ );
    if ( !solver_ )
        return;
    solver_->run( guide, normals, points, maxInitialDistSq );
}

} //namespace MR
