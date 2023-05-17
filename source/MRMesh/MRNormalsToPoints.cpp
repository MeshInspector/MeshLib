#include "MRNormalsToPoints.h"
#include "MRMesh.h"
#include "MRBitSetParallelFor.h"
#include "MRTriMath.h"
#include "MRTimer.h"

#pragma warning(push)
#pragma warning(disable: 4068) // unknown pragmas
#pragma warning(disable: 5054) // operator '|': deprecated between enumerations of different types
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#pragma clang diagnostic ignored "-Wunknown-warning-option" // for next one
#pragma clang diagnostic ignored "-Wunused-but-set-variable" // for newer clang
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#pragma clang diagnostic pop
#pragma warning(pop)

namespace MR
{

namespace
{

class Solver : public NormalsToPoints::ISolver
{
public:
    virtual void prepare( const MeshTopology & topology, float guideWeight ) override;
    virtual void run( const VertCoords & guide, const FaceNormals & normals, VertCoords & points ) override;

private:
    const MeshTopology * topology_ = nullptr;
    float guideWeight_ = 0;
    using SparseMatrix = Eigen::SparseMatrix<double,Eigen::RowMajor>;
    Vector<int, FaceId> face2row_;
    SparseMatrix mat_;
    Eigen::VectorXd rhs_[3];
    using SparseMatrixColMajor = Eigen::SparseMatrix<double,Eigen::ColMajor>;
    Eigen::SimplicialLDLT<SparseMatrixColMajor> ldlt_;
};

void Solver::prepare( const MeshTopology & topology, float guideWeight )
{
    MR_TIMER
    topology_ = &topology;
    guideWeight_ = guideWeight;
    std::vector< Eigen::Triplet<double> > mTriplets;
    const int nVerts = (int)topology.vertSize();
    mTriplets.reserve( nVerts + 6 * topology.numValidFaces() );
    // every point shall be close to corresponding guide point (with small weight)
    for ( int v = 0; v < nVerts; ++v )
        mTriplets.emplace_back( v, v, guideWeight );

    // add 2 equations per triangle for relative position of triangle points
    const int nRows = nVerts + 2 * topology.numValidFaces();
    int row = nVerts;
    face2row_.resize( topology.faceSize() );
    for ( auto f : topology.getValidFaces() )
    {
        face2row_[f] = row;

        VertId vs[3];
        topology.getTriVerts( f, vs );

        mTriplets.emplace_back( row, vs[0],  2 );
        mTriplets.emplace_back( row, vs[1], -1 );
        mTriplets.emplace_back( row, vs[2], -1 );
        ++row;

        mTriplets.emplace_back( row, vs[0], -1 );
        mTriplets.emplace_back( row, vs[1],  2 );
        mTriplets.emplace_back( row, vs[2], -1 );
        ++row;
    }
    assert( row == nRows );
    assert( mTriplets.size() == nVerts + 6 * topology.numValidFaces() );

    mat_.resize( nRows, nVerts );
    mat_.setFromTriplets( mTriplets.begin(), mTriplets.end() );

    SparseMatrix A = mat_.adjoint() * mat_;
    ldlt_.compute( A );

    for ( int i = 0; i < 3; ++i )
        rhs_[i].resize( nRows );
}

void Solver::run( const VertCoords & guide, const FaceNormals & normals, VertCoords & points )
{
    MR_TIMER
    assert( topology_ );
    if ( !topology_ )
        return;

    const int nVerts = (int)topology_->vertSize();
    // every point shall be close to corresponding guide point (with small weight)
    tbb::parallel_for( tbb::blocked_range( 0_v, VertId( nVerts ) ), [&] ( const tbb::blocked_range<VertId>& range )
    {
        for ( VertId v = range.begin(); v < range.end(); ++v )
        {
            for ( int i = 0; i < 3; ++i )
                rhs_[i][v] = guideWeight_ * guide[v][i];
        }
    } );

    // add 2 equations per triangle for relative position of projected triangle points
    BitSetParallelFor( topology_->getValidFaces(), [&]( FaceId f )
    {
        VertId vs[3];
        topology_->getTriVerts( f, vs );
        const auto projectedTri = triangleWithNormal( { points[vs[0]], points[vs[1]], points[vs[2]], }, normals[f] );
        const auto d0 = 2.0f * projectedTri[0] - projectedTri[1] - projectedTri[2];
        const auto d1 = 2.0f * projectedTri[1] - projectedTri[0] - projectedTri[2];
        const int row = face2row_[f];
        for ( int i = 0; i < 3; ++i )
        {
            rhs_[i][row    ] =  d0[i];
            rhs_[i][row + 1] =  d1[i];
        }
    } );

    // solve linear equations
    Eigen::VectorXd sol[3];
    tbb::parallel_for( tbb::blocked_range<int>( 0, 3 ), [&]( const tbb::blocked_range<int> & range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
            sol[i] = ldlt_.solve( mat_.adjoint() * rhs_[i] );
    } );

    // copy back the solution into points
    tbb::parallel_for( tbb::blocked_range( 0_v, VertId( nVerts ) ), [&] ( const tbb::blocked_range<VertId>& range )
    {
        for ( VertId v = range.begin(); v < range.end(); ++v )
        {
            for ( int i = 0; i < 3; ++i )
                points[v][i] = (float)sol[i][v];
        }
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
    assert( solver_ );
    if ( !solver_ )
        return;
    solver_->run( guide, normals, points );
}

} //namespace MR
