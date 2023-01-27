#include "MRBestFit.h"
#include "MRPlane3.h"
#include "MRLine3.h"
#include "MRAffineXf3.h"
#include "MRMesh.h"
#include "MRGTest.h"
#include <cassert>

// unknown pragmas
#pragma warning(disable:4068)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#include <Eigen/Eigenvalues>
#pragma clang diagnostic pop

namespace MR
{
 
void PointAccumulator::addPoint( const Vector3d & pt )
{
    sumWeight_ += 1;
    momentum1_ += pt;
    momentum2_ += outer( pt, pt );
}

void PointAccumulator::addPoint( const Vector3d & pt, double weight )
{
    sumWeight_ += weight;
    auto wpt = weight * pt;
    momentum1_ += wpt;
    momentum2_ += outer( wpt, pt );
}

bool PointAccumulator::getCenteredCovarianceEigen( Vector3d & centroid, Matrix3d & eigenvectors, Vector3d & eigenvalues ) const
{
    if ( sumWeight_ <= 0 )
    {
        assert( false );
        return false;
    }

    double rW = 1.0 / sumWeight_;
    centroid = rW * momentum1_;

    // covariance matrix relative to centroid
    auto cov = momentum2_ - outer( rW * momentum1_,  momentum1_ );

    static_assert( sizeof( Vector3d ) == sizeof( Eigen::Vector3d ), "types have distinct memory layout" );
    static_assert( sizeof( Matrix3d ) == sizeof( Eigen::Matrix3d ), "types have distinct memory layout" );

    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver( Eigen::Map<Eigen::Matrix3d>{ &cov.x.x } );

    // columns in column-major storage format of Eigen are converted into rows of Matrix3d
    Eigen::Map<Eigen::Matrix3d>{ &eigenvectors.x.x } = solver.eigenvectors();
    Eigen::Map<Eigen::Vector3d>{ &eigenvalues.x } = solver.eigenvalues();

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

Vector3d PlaneAccumulator::findBestCrossPoint( const Vector3d & p0, double tol ) const
{
    return p0 + mat_.solve( rhs_ - mat_ * p0, tol );
}

} //namespace MR
