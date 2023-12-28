#include "MRBestFit.h"
#include "MRPlane3.h"
#include "MRLine3.h"
#include "MRAffineXf3.h"
#include "MRMesh.h"
#include "MRGTest.h"
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

void accumulateFaceCenters( PointAccumulator& accum, const MeshPart& mp, const AffineXf3f* xf /*= nullptr */ )
{
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

void PlaneAccumulator::addPlane( const Plane3d & pl )
{
    mat_ += outerSquare( pl.n );
    rhs_ += pl.d * pl.n;
}

Vector3d PlaneAccumulator::findBestCrossPoint( const Vector3d & p0, double tol, int * rank, Vector3d * space ) const
{
    return p0 + mat_.solve( rhs_ - mat_ * p0, tol, rank, space );
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
