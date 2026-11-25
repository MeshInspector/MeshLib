#include "MRBestFit.h"
#include "MRPlane3.h"
#include "MRLine3.h"
#include "MRAffineXf3.h"
#include "MRMesh.h"
#include "MRPointCloud.h"
#include "MRTimer.h"
#include "MRPolylineEdgeIterator.h"
#include "MRBox.h"
#include "MRSymMatrix2.h"
#include <MRPch/MREigenCore.h>
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4244) // conversion from double to float
#pragma warning(disable: 4464) // relative include path contains '..'
#endif
#include <unsupported/Eigen/NonLinearOptimization>
#ifdef _MSC_VER
#pragma warning(pop)
#endif
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

/// relative tolerance of the eigenvalue gaps and ranks below: SymMatrix3::eigens splits equal eigenvalues by up to about 1.5e-8 of the largest one,
/// and well-defined fits have much larger gaps, e.g. the smallest eigenvalue of the covariance of the points of a 5 degree arc is 1.4e-4 of the largest one
constexpr double cEigenTol = 1e-6;

template <typename V>
std::optional<Sphere<V>> findBestSphereT( const std::vector<V>& points )
{
    V mean;
    for ( const auto& p : points )
        mean += p;
    if ( !points.empty() )
        mean /= double( points.size() );
    // |d - c|^2 = r^2 for the centered points d = p - mean is linear in c and s = r^2 - |c|^2: |d|^2 = 2 * dot( d, c ) + s;
    // since sum d = 0, its least squares solution decouples: ( sum d d^T ) c = sum d |d|^2 / 2 and s = mean |d|^2
    decltype( outerSquare( mean ) ) m;
    V rhs;
    double s = 0;
    for ( const auto& p : points )
    {
        const V d = p - mean;
        m += outerSquare( d );
        rhs += d.lengthSq() * d;
        s += d.lengthSq();
    }
    // the center is not unique if the points are on a line (circle) or in a plane (sphere)
    int rank = 0;
    const V c = m.pseudoinverse( cEigenTol, &rank ) * rhs / 2.0;
    const double r2 = s / double( points.size() ) + c.lengthSq();
    if ( rank < V::elements || !std::isfinite( r2 ) )
        return {};
    return Sphere<V>( mean + c, std::sqrt( r2 ) );
}

/// the distances from the points to the torus with the parameters: the shift of the center from c0, not-unit axis, major and minor radii;
/// the functor for Eigen::LevenbergMarquardt
struct TorusDistances
{
    const std::vector<Vector3d>& points;
    Vector3d c0;

    Torus3d torus( const Eigen::VectorXd& x ) const
    {
        return Torus3d( c0 + Vector3d( x[0], x[1], x[2] ), Vector3d( x[3], x[4], x[5] ).normalized(), std::abs( x[6] ), std::abs( x[7] ) );
    }
    int values() const { return int( points.size() ); }
    int operator()( const Eigen::VectorXd& x, Eigen::VectorXd& f ) const
    {
        const auto t = torus( x );
        for ( int i = 0; i < values(); ++i )
            f[i] = t.distance( points[i] );
        return 0;
    }
    /// the derivatives of a distance by the center: -n, by the not-unit axis: majorRadius * dot( n, axis ) * e / |axis|,
    /// by the radii: -dot( n, e ) and -1, where n is the normal of the torus and e is the unit direction from the axis to the point
    int df( const Eigen::VectorXd& x, Eigen::MatrixXd& jac ) const
    {
        const auto t = torus( x );
        const double rLen = 1 / Vector3d( x[3], x[4], x[5] ).length();
        for ( int i = 0; i < values(); ++i )
        {
            const Vector3d d = points[i] - t.center, n = t.normal( points[i] );
            const Vector3d e = ( d - dot( d, t.axis ) * t.axis ).normalized();
            const Vector3d da = t.majorRadius * dot( n, t.axis ) * rLen * e;
            jac.row( i ) << -n.x, -n.y, -n.z, da.x, da.y, da.z, x[6] < 0 ? dot( n, e ) : -dot( n, e ), x[7] < 0 ? 1 : -1;
        }
        return 0;
    }
};

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
    // the axis has zero reciprocal product with every normal line (p, n): in Pluecker coordinates (d, m) of the axis, dot( d, p x n ) + dot( m, n ) = 0;
    // the points are centered and scaled to make both parts of the coordinates comparable, so |p x n| <= |n|
    Vector3d c;
    for ( const auto& l : normalLines )
        c += l.p;
    if ( !normalLines.empty() )
        c /= double( normalLines.size() );
    double sc = 0;
    for ( const auto& l : normalLines )
        sc = std::max( sc, ( l.p - c ).length() );
    if ( !( sc > 0 ) )
        return {};
    // for given unit d, the best m = -nn^+ b^T d with b = sum ( p x n ) n^T and nn = sum n n^T, and the cost is dot( d, s d ),
    // where s sums the outer squares of the residuals of the least squares fit of p x n by b nn^+ n
    Matrix3d b = Matrix3d::zero();
    SymMatrix3d nn;
    for ( const auto& l : normalLines )
    {
        b += outer( cross( ( l.p - c ) / sc, l.d ), l.d );
        nn += outerSquare( l.d );
    }
    int rank = 0;
    const auto nnInv = nn.pseudoinverse( cEigenTol, &rank );
    SymMatrix3d s;
    for ( const auto& l : normalLines )
        s += outerSquare( cross( ( l.p - c ) / sc, l.d ) - b * ( nnInv * l.d ) );
    Matrix3d vecs;
    const auto eig = s.eigens( &vecs );
    // the direction is not unique if the two smallest eigenvalues of s are close relative to the trace of nn, which bounds them:
    // e.g. all lines through the center of a sphere cross its normals;
    // the position is not unique if nn has rank 1: e.g. any line along the normal of a plane crosses its normals
    if ( !( eig[1] - eig[0] > cEigenTol * nn.trace() ) || rank < 2 )
        return {};
    const Vector3d d = vecs.x;
    Vector3d m = -( nnInv * ( b.transposed() * d ) );
    m -= dot( m, d ) * d;
    return Line3d( c + sc * cross( d, m ), d );
}

std::optional<Cylinder3d> findBestCylinderByNormals( const std::vector<Vector3d>& points, const std::vector<Vector3d>& normals )
{
    SymMatrix3d nn;
    for ( const auto& n : normals )
        nn += outerSquare( n );
    Matrix3d vecs;
    const auto eig = nn.eigens( &vecs );
    // the axis is not unique if the two smallest eigenvalues are close, e.g. for the normals of a plane or of a sphere cap
    if ( !( eig[1] - eig[0] > cEigenTol * eig[2] ) )
        return {};
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
    // the generator of the cone fitted orthogonally to the points (h, rho): the position along the axis and the distance to it
    const Vector3d a = axis.d.normalized();
    PointAccumulator acc;
    for ( const auto& p : points )
    {
        const Vector3d d = p - axis.p;
        const double h = dot( d, a );
        acc.addPoint( Vector3d( h, ( d - h * a ).length(), 0 ) );
    }
    if ( !acc.valid() )
        return {};
    const auto l = acc.getBestLine();
    // a constant distance (a cylinder) or a constant position (a plane) gives no cone; so do coincident points: then the line goes along z
    const double angle = std::atan2( std::abs( l.d.y ), std::abs( l.d.x ) );
    if ( !( angle > 1e-6 && angle < PI / 2 - 1e-6 ) )
        return {};
    return coneSpanning( points, Line3d( axis.p + a * ( l.p.x - l.p.y * l.d.x / l.d.y ), l.d.x * l.d.y > 0 ? a : -a ), angle );
}

std::optional<Cone3d> findBestConeByTangentPlanes( const std::vector<Vector3d>& points, const std::vector<Plane3d>& tangentPlanes )
{
    // all unit normals make the same angle with the axis a: their tips are in the plane dot( a, x ) = k
    PointAccumulator tips;
    // the apex is on all tangent planes
    PlaneAccumulator apexAcc;
    for ( const auto& plane : tangentPlanes )
    {
        const auto pl = plane.normalized();
        if ( pl.n == Vector3d{} )
            continue; // e.g. the plane of a degenerate triangle
        tips.addPoint( pl.n );
        apexAcc.addPlane( pl );
    }
    if ( !tips.valid() )
        return {};
    Vector3d tipsCenter, eig;
    Matrix3d vecs;
    tips.getCenteredCovarianceEigen( tipsCenter, vecs, eig );
    // the tips do not define the plane if they are on a line (two distinct normals) or coincide (parallel planes);
    // k is about zero if the planes are tangent to a cylinder; to tell it reliably, the two smallest eigenvalues must differ by 1e-4 of the largest one,
    // since SymMatrix3::eigens finds the eigenvector of the smallest one with the error growing as eps / gap^2 (the tips of about 4.5 degrees of a revolution pass)
    const Vector3d a = vecs.x;
    if ( !( eig[1] - eig[0] > 1e-4 * eig[2] ) || !( std::abs( dot( a, tipsCenter ) ) > 1e-6 ) )
        return {};
    Vector3d center;
    for ( const auto& p : points )
        center += p;
    int rank = 0;
    const Vector3d apex = apexAcc.findBestCrossPoint( center / double( points.size() ), std::numeric_limits<double>::epsilon(), &rank );
    if ( rank < 3 )
        return {};
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
    if ( !( angle > 1e-6 ) )
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
    // the center is shifted from the given one, so the relative tolerances of the minimization do not depend on the distance to the origin;
    // the minimization accepts only the steps decreasing the sum of squared distances
    TorusDistances f{ points, torus.center };
    Eigen::VectorXd x( 8 );
    x << 0, 0, 0, torus.axis.x, torus.axis.y, torus.axis.z, torus.majorRadius, torus.minorRadius;
    Eigen::LevenbergMarquardt<TorusDistances> lm( f );
    lm.minimize( x );
    const auto res = f.torus( x );
    return std::isfinite( res.center.x + res.center.y + res.center.z + res.axis.x + res.axis.y + res.axis.z + res.majorRadius + res.minorRadius ) ? res : torus;
}

} //namespace MR
