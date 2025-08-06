#include "MRBestFit.h"
#include "MRPlane3.h"
#include "MRLine3.h"
#include "MRAffineXf3.h"
#include "MRMesh.h"
#include "MRPointCloud.h"
#include "MRGTest.h"
#include "MRTimer.h"
#include "MRPolylineEdgeIterator.h"
#include <cassert>

namespace MR
{

void PointAccumulator::addPoint( const Vector3d & pt )
{
    sumWeight_ += 1;
    momentum1_ += pt;
    momentum2_ += outerSquare( pt );
}

void PointAccumulator::addPoint( const Vector3d & pt, double weight )
{
    sumWeight_ += weight;
    momentum1_ += weight * pt;
    momentum2_ += weight * outerSquare( pt );
}

bool PointAccumulator::getCenteredCovarianceEigen( Vector3d & centroid, Matrix3d & eigenvectors, Vector3d & eigenvalues ) const
{
    if ( sumWeight_ <= 0 )
    {
        assert( false );
        return false;
    }

    const double rW = 1.0 / sumWeight_;
    centroid = rW * momentum1_;

    // covariance matrix relative to centroid
    const auto cov = momentum2_ - rW * outerSquare( momentum1_ );
    eigenvalues = cov.eigens( &eigenvectors );
    return true;
}

bool PointAccumulator::getCenteredCovarianceEigen( Vector3f& centroid, Matrix3f& eigenvectors, Vector3f& eigenvalues ) const
{
    Vector3d centroidd;
    Matrix3d eigenvectorsd;
    Vector3d eigenvaluesd;
    bool res = getCenteredCovarianceEigen( centroidd, eigenvectorsd, eigenvaluesd );
    centroid = Vector3f( centroidd );
    eigenvectors = Matrix3f( eigenvectorsd );
    eigenvalues = Vector3f( eigenvaluesd );
    return res;
}

AffineXf3d PointAccumulator::getBasicXf() const
{
    Vector3d centroid;
    Matrix3d eigenvectors;
    Vector3d eigenvalues;
    [[maybe_unused]] bool ok = getCenteredCovarianceEigen( centroid, eigenvectors, eigenvalues );
    assert( ok );

    AffineXf3d res;
    res.b = centroid;
    if ( mixed( eigenvectors.x, eigenvectors.y, eigenvectors.z ) < 0.0f )
        eigenvectors.z = -eigenvectors.z;
    res.A = eigenvectors.transposed();
    return res;
}

std::array<AffineXf3d, 4> PointAccumulator::get4BasicXfs() const
{
    Vector3d centroid;
    Matrix3d eigenvectors;
    Vector3d eigenvalues;
    [[maybe_unused]] bool ok = getCenteredCovarianceEigen( centroid, eigenvectors, eigenvalues );
    assert( ok );

    std::array<AffineXf3d, 4> res;
    res[0].b = res[1].b = res[2].b = res[3].b = centroid;

    const auto exy = cross( eigenvectors.x, eigenvectors.y );
    res[0].A = Matrix3d::fromColumns(  eigenvectors.x,  eigenvectors.y,  exy );
    res[1].A = Matrix3d::fromColumns(  eigenvectors.x, -eigenvectors.y, -exy );
    res[2].A = Matrix3d::fromColumns( -eigenvectors.x,  eigenvectors.y, -exy );
    res[3].A = Matrix3d::fromColumns( -eigenvectors.x, -eigenvectors.y,  exy );

    return res;
}

std::array<AffineXf3f, 4> PointAccumulator::get4BasicXfs3f() const
{
    const auto ds = get4BasicXfs();
    return { AffineXf3f( ds[0] ), AffineXf3f( ds[1] ), AffineXf3f( ds[2] ), AffineXf3f( ds[3] ) };
}

Plane3d PointAccumulator::getBestPlane() const
{
    Vector3d centroid;
    Matrix3d eigenvectors;
    Vector3d eigenvalues;
    if ( !getCenteredCovarianceEigen( centroid, eigenvectors, eigenvalues ) )
        return {};

    return Plane3d::fromDirAndPt( eigenvectors.x, centroid );
}

Line3d PointAccumulator::getBestLine() const
{
    Vector3d centroid;
    Matrix3d eigenvectors;
    Vector3d eigenvalues;
    if ( !getCenteredCovarianceEigen( centroid, eigenvectors, eigenvalues ) )
        return {};

    return { centroid, eigenvectors.z };
}

void accumulatePoints( PointAccumulator& accum, const std::vector<Vector3f>& points, const AffineXf3f* xf )
{
    MR_TIMER;
    for ( const auto& p : points )
        accum.addPoint( p.transformed( xf ) );
}

void accumulateWeighedPoints( PointAccumulator& accum, const std::vector<Vector3f>& points, const std::vector<float>& weights, const AffineXf3f* xf )
{
    MR_TIMER;
    assert( points.size() == weights.size() );
    for ( auto i = 0; i < points.size(); ++i )
        accum.addPoint( points[i].transformed( xf ), weights[i] );
}

void accumulateFaceCenters( PointAccumulator& accum, const MeshPart& mp, const AffineXf3f* xf /*= nullptr */ )
{
    MR_TIMER;
    const auto& topology = mp.mesh.topology;
    const auto& edgePerFaces = topology.edgePerFace();
    const auto& faceIds = topology.getFaceIds( mp.region );
    for ( auto f : faceIds )
    {
        if ( mp.region && !topology.hasFace( f ) )
            continue; // skip region-faces, which does not actually exist
        auto edge = edgePerFaces[f];
        if ( edge.valid() )
        {
            VertId v0, v1, v2;
            topology.getLeftTriVerts( edge, v0, v1, v2 );
            //area of triangle corresponds to the weight of each point
            float triArea = mp.mesh.leftDirDblArea( edge ).length();
            auto center = ( 1 / 3.0f ) * Vector3f{ mp.mesh.points[v0] + mp.mesh.points[v1] + mp.mesh.points[v2] };
            accum.addPoint( center.transformed( xf ), triArea );
        }
    }
}

void accumulateLineCenters( PointAccumulator& accum, const Polyline3& pl, const AffineXf3f* xf )
{
    MR_TIMER;
    const auto& topology = pl.topology;
    for ( auto edge : undirectedEdges(topology) )
    {
        const auto& p1 = pl.orgPnt( edge );
        const auto& p2 = pl.destPnt( edge );
        auto center = (p1 + p2) / 2.0f;
        const float length = ( p1 - p2 ).length();
        accum.addPoint( center.transformed( xf ), length );
    }
}

void accumulatePoints( PointAccumulator& accum, const PointCloudPart& pcp, const AffineXf3f* xf )
{
    MR_TIMER;
    for ( auto v : pcp.cloud.getVertIds( pcp.region ) )
        accum.addPoint( pcp.cloud.points[v].transformed( xf ) );
}

void PlaneAccumulator::addPlane( const Plane3d & pl )
{
    mat_ += outerSquare( pl.n );
    rhs_ += pl.d * pl.n;
}

Vector3d PlaneAccumulator::findBestCrossPoint( const Vector3d & p0, double tol, int * rank, Vector3d * space ) const
{
    return p0 + mat_.pseudoinverse( tol, rank, space ) * ( rhs_ - mat_ * p0 );
}

Vector3f PlaneAccumulator::findBestCrossPoint( const Vector3f & p0, float tol, int * rank, Vector3f * space ) const
{
    Vector3d mySpace;
    Vector3f res{ findBestCrossPoint( Vector3d{ p0 }, tol, rank, space ? &mySpace : nullptr ) };
    if ( space )
        *space = Vector3f( mySpace );
    return res;
}

} //namespace MR
