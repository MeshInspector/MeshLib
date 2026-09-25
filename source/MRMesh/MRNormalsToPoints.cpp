#include "MRNormalsToPoints.h"
#include "MRMesh.h"
#include "MRRegionBoundary.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRTriMath.h"
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
    virtual void prepare( const MeshTopology & topology, float guideWeight, const VertBitSet * region ) override;
    virtual void run( const VertCoords & guide, const FaceNormals & normals, VertCoords & points, float maxInitialDistSq ) override;

private:
    const MeshTopology * topology_ = nullptr;
    float guideWeight_ = 0;
    VertBitSet verts_; // vertices to move
    FaceBitSet faces_; // faces with at least one vertex to move
    Vector<int, VertId> vert2col_; // -1 for fixed vertices
    using SparseMatrix = Eigen::SparseMatrix<double,Eigen::RowMajor>;
    Vector<int, FaceId> face2row_;
    SparseMatrix mat_;
    Eigen::VectorXd rhs_[3];
    using SparseMatrixColMajor = Eigen::SparseMatrix<double,Eigen::ColMajor>;
    Eigen::SimplicialLDLT<SparseMatrixColMajor> ldlt_;
};

void Solver::prepare( const MeshTopology & topology, float guideWeight, const VertBitSet * region )
{
    MR_TIMER;
    topology_ = &topology;
    // the equations per triangle below are ~3 times lighter than centered ones (2*p0-p1-p2, 2*p1-p0-p2),
    // so the guide weight is reduced by sqrt(3) to keep the same balance
    guideWeight_ = guideWeight / std::sqrt( 3.0f );
    verts_ = topology.getVertIds( region );
    faces_ = region ? getIncidentFaces( topology, verts_ ) : topology.getValidFaces();
    vert2col_.clear();
    vert2col_.resize( topology.vertSize(), -1 );
    int nCols = 0;
    for ( auto v : verts_ )
        vert2col_[v] = nCols++;
    const int nFaces = (int)faces_.count();

    std::vector< Eigen::Triplet<double> > mTriplets;
    mTriplets.reserve( nCols + 4 * nFaces );
    // every point shall be close to corresponding guide point (with small weight)
    for ( int c = 0; c < nCols; ++c )
        mTriplets.emplace_back( c, c, guideWeight_ );

    // add 2 equations per triangle for relative position of triangle points: p0-p1 and p0-p2,
    // fixed vertices go to the right-hand side in run()
    const int nRows = nCols + 2 * nFaces;
    int row = nCols;
    face2row_.resize( topology.faceSize() );
    for ( auto f : faces_ )
    {
        face2row_[f] = row;

        VertId vs[3];
        topology.getTriVerts( f, vs );
        const auto add = [&]( VertId v, double x )
        {
            if ( const int c = vert2col_[v]; c >= 0 )
                mTriplets.emplace_back( row, c, x );
        };

        add( vs[0],  1 );
        add( vs[1], -1 );
        ++row;

        add( vs[0],  1 );
        add( vs[2], -1 );
        ++row;
    }
    assert( row == nRows );

    mat_.resize( nRows, nCols );
    mat_.setFromTriplets( mTriplets.begin(), mTriplets.end() );

    SparseMatrix A = mat_.adjoint() * mat_;
    ldlt_.compute( A );

    for ( int i = 0; i < 3; ++i )
        rhs_[i].resize( nRows );
}

void Solver::run( const VertCoords & guide, const FaceNormals & normals, VertCoords & points, float maxInitialDistSq )
{
    MR_TIMER;
    assert( topology_ );
    if ( !topology_ )
        return;

    // every point shall be close to corresponding guide point (with small weight)
    BitSetParallelFor( verts_, [&]( VertId v )
    {
        const int c = vert2col_[v];
        for ( int i = 0; i < 3; ++i )
            rhs_[i][c] = guideWeight_ * guide[v][i];
    } );

    // add 2 equations per triangle for relative position of projected triangle points
    BitSetParallelFor( faces_, [&]( FaceId f )
    {
        VertId vs[3];
        topology_->getTriVerts( f, vs );
        const auto projectedTri = triangleWithNormal( { points[vs[0]], points[vs[1]], points[vs[2]], }, normals[f] );
        auto d0 = projectedTri[0] - projectedTri[1];
        auto d1 = projectedTri[0] - projectedTri[2];
        // move fixed vertices to the right-hand side
        if ( vert2col_[vs[0]] < 0 )
        {
            d0 -= points[vs[0]];
            d1 -= points[vs[0]];
        }
        if ( vert2col_[vs[1]] < 0 )
            d0 += points[vs[1]];
        if ( vert2col_[vs[2]] < 0 )
            d1 += points[vs[2]];
        const int row = face2row_[f];
        for ( int i = 0; i < 3; ++i )
        {
            rhs_[i][row    ] =  d0[i];
            rhs_[i][row + 1] =  d1[i];
        }
    } );

    // solve linear equations
    Eigen::VectorXd sol[3];
    ParallelFor( 0, 3, [&]( int i )
    {
        sol[i] = ldlt_.solve( mat_.adjoint() * rhs_[i] );
    } );

    // copy back the solution into points
    const bool limitNearInitial = std::isfinite( maxInitialDistSq );
    BitSetParallelFor( verts_, [&]( VertId v )
    {
        const int c = vert2col_[v];
        Vector3f np;
        for ( int i = 0; i < 3; ++i )
            np[i] = (float)sol[i][c];
        if ( limitNearInitial )
            np = getLimitedPos( np, guide[v], maxInitialDistSq );
        points[v] = np;
    } );
}

} //anonymous namespace

void NormalsToPoints::prepare( const MeshTopology & topology, float guideWeight, const VertBitSet * region )
{
    solver_ = std::make_unique<Solver>();
    solver_->prepare( topology, guideWeight, region );
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
