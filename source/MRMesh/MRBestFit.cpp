#include "MRBestFit.h"
#include "MRPlane3.h"
#include "MRLine3.h"
#include "MRAffineXf3.h"
#include "MRMesh.h"
#include "MRPointCloud.h"
#include "MRTimer.h"
#include "MRPolylineEdgeIterator.h"
#include "MRBox.h"
#include <MRPch/MREigenCore.h>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
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

namespace
{

template <typename V>
std::optional<Sphere<V>> findBestSphereT( const std::vector<V>& points )
{
    constexpr int n = V::elements;
    using EV = Eigen::Matrix<double, n + 1, 1>;
    V mean;
    for ( const auto& p : points )
        mean += p;
    if ( !points.empty() )
        mean /= double( points.size() );
    // |p - c|^2 = r^2  <=>  |p|^2 = 2*dot( p, c ) + r^2 - |c|^2
    Eigen::Matrix<double, n + 1, n + 1> m = Eigen::Matrix<double, n + 1, n + 1>::Zero();
    EV rhs = EV::Zero();
    for ( const auto& p : points )
    {
        const V d = p - mean;
        EV row;
        for ( int i = 0; i < n; ++i )
            row[i] = 2 * d[i];
        row[n] = 1;
        m += row * row.transpose();
        rhs += row * d.lengthSq();
    }
    // the rank-revealing solver gives the stablest result for a short arc, where the matrix is nearly degenerate
    const EV sol = m.completeOrthogonalDecomposition().solve( rhs );
    V c;
    for ( int i = 0; i < n; ++i )
        c[i] = sol[i];
    const double r2 = sol[n] + c.lengthSq();
    if ( !( r2 > 0 ) || !std::isfinite( r2 ) )
        return {};
    return Sphere<V>( mean + c, std::sqrt( r2 ) );
}

/// the cone with given apex, axis and angle, and the height spanning the points
Cone3d coneSpanning( const std::vector<Vector3d>& points, const Line3d& axis, double angle )
{
    double height = 0;
    for ( const auto& p : points )
        height = std::max( height, std::abs( dot( p - axis.p, axis.d ) ) );
    return Cone3d( axis, angle, height );
}

} //anonymous namespace

std::optional<Sphere3d> findBestSphere( const std::vector<Vector3d>& points )
{
    return findBestSphereT( points );
}

std::optional<Sphere2d> findBestCircle( const std::vector<Vector2d>& points )
{
    return findBestSphereT( points );
}

std::optional<Line3d> findRevolutionAxis( const std::vector<Line3d>& normalLines )
{
    // the axis has zero reciprocal product with every normal line: in Pluecker coordinates (d, m) of the axis, dot( d, p x n ) + dot( m, n ) = 0;
    // the points are scaled to make both parts of the coordinates comparable
    double sc = 1e-9;
    for ( const auto& l : normalLines )
        sc = std::max( { sc, std::abs( l.p.x ), std::abs( l.p.y ), std::abs( l.p.z ) } );
    Eigen::Matrix<double, 6, 6> m = Eigen::Matrix<double, 6, 6>::Zero();
    for ( const auto& l : normalLines )
    {
        const Vector3d mo = cross( l.p / sc, l.d );
        Eigen::Matrix<double, 6, 1> row;
        row << mo.x, mo.y, mo.z, l.d.x, l.d.y, l.d.z;
        m += row * row.transpose();
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es( m );
    const auto v = es.eigenvectors().col( 0 );
    Vector3d d( v[0], v[1], v[2] ), mom( v[3], v[4], v[5] );
    const double nd = d.length();
    if ( !( nd > 1e-6 ) )
        return {};
    d /= nd;
    mom /= nd;
    mom -= d * dot( mom, d );
    return Line3d( cross( d, mom ) * sc, d );
}

std::optional<Cylinder3d> findBestCylinderByNormals( const std::vector<Vector3d>& points, const std::vector<Vector3d>& normals )
{
    SymMatrix3d nn;
    for ( const auto& n : normals )
        nn += outerSquare( n );
    Matrix3d vecs;
    nn.eigens( &vecs );
    const Vector3d a = vecs.x;
    const auto [u, v] = a.perpendicular();
    std::vector<Vector2d> proj( points.size() );
    for ( size_t i = 0; i < points.size(); ++i )
        proj[i] = Vector2d( dot( points[i], u ), dot( points[i], v ) );
    const auto circle = findBestCircle( proj );
    if ( !circle )
        return {};
    const Vector3d c = circle->center.x * u + circle->center.y * v;
    MinMaxd range;
    for ( const auto& p : points )
        range.include( dot( p - c, a ) );
    return Cylinder3d( c + range.center() * a, a, circle->radius, range.size() );
}

std::optional<Cone3d> findBestConeWithAxis( const std::vector<Vector3d>& points, const Line3d& axis )
{
    // the distance to the axis: rho = k * h + b
    const Vector3d a = axis.d.normalized();
    double sh = 0, shh = 0, sr = 0, shr = 0;
    for ( const auto& p : points )
    {
        const Vector3d d = p - axis.p;
        const double h = dot( d, a );
        const double rho = ( d - h * a ).length();
        sh += h;
        shh += h * h;
        sr += rho;
        shr += h * rho;
    }
    const double n = double( points.size() );
    const double det = shh * n - sh * sh;
    if ( std::abs( det ) < 1e-30 )
        return {};
    const double k = ( shr * n - sh * sr ) / det;
    const double b = ( shh * sr - sh * shr ) / det;
    if ( std::abs( k ) < 1e-6 )
        return {};
    return coneSpanning( points, Line3d( axis.p - a * ( b / k ), k > 0 ? a : -a ), std::atan( std::abs( k ) ) );
}

std::optional<Cone3d> findBestConeByTangentPlanes( const std::vector<Vector3d>& points, const std::vector<Plane3d>& tangentPlanes )
{
    // all normals make the same angle with the axis: dot( n, a ) = k
    Eigen::Matrix4d m = Eigen::Matrix4d::Zero();
    // the apex is on all tangent planes
    PlaneAccumulator apexAcc;
    for ( const auto& plane : tangentPlanes )
    {
        const auto pl = plane.normalized();
        const Eigen::Vector4d row( pl.n.x, pl.n.y, pl.n.z, -1 );
        m += row * row.transpose();
        apexAcc.addPlane( pl );
    }
    const Eigen::Vector4d v = Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d>( m ).eigenvectors().col( 0 );
    Vector3d a( v[0], v[1], v[2] );
    const double len = a.length();
    if ( !( len > 1e-9 ) )
        return {};
    a /= len;
    const Vector3d apex = apexAcc.findBestCrossPoint( Vector3d{}, std::numeric_limits<double>::epsilon() );
    double sumH = 0, sumHH = 0, sumHR = 0;
    for ( const auto& p : points )
    {
        const Vector3d d = p - apex;
        const double h = dot( d, a );
        sumH += h;
        sumHH += h * h;
        sumHR += std::abs( h ) * ( d - h * a ).length();
    }
    if ( !( sumHH > 0 ) )
        return {};
    const double angle = std::atan( sumHR / sumHH );
    if ( !std::isfinite( apex.x + apex.y + apex.z ) || !( angle > 1e-6 ) )
        return {};
    return coneSpanning( points, Line3d( apex, sumH < 0 ? -a : a ), angle );
}

std::optional<Torus3d> findBestTorusWithAxis( const std::vector<Vector3d>& points, const Line3d& axis )
{
    const Vector3d a = axis.d.normalized();
    std::vector<Vector2d> rh( points.size() );
    for ( size_t i = 0; i < points.size(); ++i )
    {
        const Vector3d d = points[i] - axis.p;
        const double h = dot( d, a );
        rh[i] = Vector2d( ( d - h * a ).length(), h );
    }
    const auto circle = findBestCircle( rh );
    if ( !circle || !( circle->center.x > 0 ) )
        return {};
    return Torus3d( axis.p + a * circle->center.y, a, circle->center.x, circle->radius );
}

Torus3d refineTorus( const std::vector<Vector3d>& points, const Torus3d& torus )
{
    constexpr int np = 8;
    using EV = Eigen::Matrix<double, np, 1>;
    const auto unpack = []( const EV& x )
    {
        return Torus3d( Vector3d( x[0], x[1], x[2] ), Vector3d( x[3], x[4], x[5] ).normalized(), x[6], std::abs( x[7] ) );
    };
    const int m = int( points.size() );
    const auto residuals = [&]( const EV& x, Eigen::VectorXd& f )
    {
        const auto t = unpack( x );
        for ( int i = 0; i < m; ++i )
            f[i] = t.distance( points[i] );
    };
    EV x;
    x << torus.center.x, torus.center.y, torus.center.z, torus.axis.x, torus.axis.y, torus.axis.z, torus.majorRadius, torus.minorRadius;
    Eigen::VectorXd f( m ), f1( m );
    Eigen::Matrix<double, Eigen::Dynamic, np> jac( m, np );
    residuals( x, f );
    double cost = f.squaredNorm();
    if ( !std::isfinite( cost ) )
        return torus;
    double lambda = 1e-3;
    for ( int it = 0; it < 60; ++it )
    {
        for ( int j = 0; j < np; ++j )
        {
            EV xh = x;
            const double h = 1e-7 * std::max( 1.0, std::abs( x[j] ) );
            xh[j] += h;
            residuals( xh, f1 );
            jac.col( j ) = ( f1 - f ) / h;
        }
        const Eigen::Matrix<double, np, np> jtj = jac.transpose() * jac;
        const EV g = jac.transpose() * f;
        bool improved = false;
        for ( int k = 0; k < 12; ++k )
        {
            Eigen::Matrix<double, np, np> lhs = jtj;
            for ( int j = 0; j < np; ++j )
                lhs( j, j ) += lambda * std::max( jtj( j, j ), 1e-12 );
            const EV xn = x + lhs.ldlt().solve( -g );
            residuals( xn, f1 );
            const double c1 = f1.squaredNorm();
            if ( std::isfinite( c1 ) && c1 < cost )
            {
                const bool converged = cost - c1 <= 1e-14 * cost;
                x = xn;
                f = f1;
                cost = c1;
                lambda = std::max( lambda * 0.2, 1e-12 );
                improved = !converged;
                break;
            }
            lambda *= 5;
        }
        if ( !improved )
            break;
    }
    const auto res = unpack( x );
    return std::isfinite( res.center.x + res.center.y + res.center.z + res.axis.x + res.axis.y + res.axis.z + res.majorRadius + res.minorRadius ) ? res : torus;
}

} //namespace MR
