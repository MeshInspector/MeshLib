#include "MRMeshToStepData.hpp"
#include "MRBox.h"
#include "MRMatrix3.h"
#include "MRSymMatrix3.h"
#include "MRTimer.h"
#include "MRPch/MRFmt.h"

#include <Eigen/Dense>

#include <map>
#include <numeric>
#include <set>
#include <utility>

namespace MR
{

namespace MeshToStep
{

namespace
{

/// a curve of the boundary between two regions
struct Curve
{
    enum class Kind { Line, Circle, BSpline } kind = Kind::Line;
    V3 c, n, x;   ///< circle: center, normal and the direction to the start point
    double r = 0; ///< circle radius
    int deg = 0;  ///< B-spline degree
    std::vector<double> knots;
    std::vector<V3> cp;
    double dev = 0; ///< maximum distance from B-spline to the surfaces
};

/// a part of the boundary between two regions: one curve between two vertices
struct BRepEdge
{
    std::vector<int> verts; ///< the chain of mesh vertices
    Curve curve;
};

/// creates boundary representation of the regions, and writes it in STEP format
class BRepWriter
{
public:
    BRepWriter( const Data& d, const std::string& name ) : d_( d ), name_( name ) {}
    MeshToStepStats write( std::ostream& out );

private:
    using Surfs = std::vector<const Surf*>;
    using DirEdge = std::pair<int, int>;
    struct EdgeUse
    {
        int edge = -1;
        bool forward = true;
        bool first = false; ///< it is the first mesh edge of BRepEdge passed in this direction
    };

    double curveTol_( const Surfs& ss ) const;
    double onSurfs_( const std::vector<V3>& pts, const Surfs& ss ) const;
    std::vector<V3> pointsOf_( const std::vector<int>& ids ) const;
    std::optional<Curve> tryLine_( const std::vector<V3>& p, const Surfs& ss ) const;
    std::optional<Curve> circleFrom_( const std::vector<V3>& p, const V3& ctr, V3 n, double r, const Surfs& ss, bool closed ) const;
    std::optional<Curve> axisCircle_( const std::vector<V3>& p, const Surfs& ss, bool closed ) const;
    std::optional<Curve> tryCircle_( const std::vector<V3>& p, const Surfs& ss, bool closed ) const;
    Curve spline_( const std::vector<V3>& p, const Surfs& ss ) const;
    void splitFit_( const std::vector<int>& ids, const Surfs& ss );
    void findEdges_();
    void findLoops_();

    std::string add_( std::string s );
    std::string point_( const V3& p );
    std::string dir_( const V3& d );
    std::string axis2_( const V3& p, const V3& z, const V3* x = nullptr );
    std::string vertex_( int v );

    const Data& d_;
    std::string name_;
    std::map<DirEdge, std::pair<int, int>> bnd_; ///< boundary mesh edge (u,v) -> its left face and the index of the edge in it
    std::vector<BRepEdge> edges_;
    std::map<DirEdge, EdgeUse> edgeOf_;
    std::vector<std::vector<std::vector<std::pair<int, bool>>>> loops_; ///< region -> loops -> (edge, forward)
    std::vector<std::string> lines_;
    std::map<int, std::string> vertexIds_;
};

double BRepWriter::curveTol_( const Surfs& ss ) const
{
    double res = d_.tol;
    for ( auto s : ss )
        if ( s->type == SurfType::BSpline )
            res = std::max( res, 3 * s->bs->err );
    return res;
}

double BRepWriter::onSurfs_( const std::vector<V3>& pts, const Surfs& ss ) const
{
    double res = 0;
    for ( auto s : ss )
        res = std::max( res, s->maxDist( pts ) );
    return res;
}

std::vector<V3> BRepWriter::pointsOf_( const std::vector<int>& ids ) const
{
    std::vector<V3> res;
    res.reserve( ids.size() );
    for ( int v : ids )
        res.push_back( d_.pts[v] );
    return res;
}

std::optional<Curve> BRepWriter::tryLine_( const std::vector<V3>& p, const Surfs& ss ) const
{
    V3 dir = p.back() - p.front();
    const double len = dir.length();
    if ( len < 1e-9 * d_.diag )
        return {};
    dir /= len;
    for ( const auto& q : p )
    {
        const V3 r = q - p.front();
        if ( ( r - dir * dot( r, dir ) ).length() > 0.5 * d_.tol )
            return {};
    }
    std::vector<V3> samples;
    for ( int i = 0; i <= 8; ++i )
        samples.push_back( p.front() + ( p.back() - p.front() ) * ( i / 8.0 ) );
    if ( onSurfs_( samples, ss ) > curveTol_( ss ) )
        return {};
    return Curve{};
}

std::optional<Curve> BRepWriter::circleFrom_( const std::vector<V3>& p, const V3& ctr, V3 n, double r, const Surfs& ss, bool closed ) const
{
    // orient the axis so that the chain runs counter-clockwise
    V3 tot;
    for ( size_t i = 0; i + 1 < p.size(); ++i )
        tot += cross( p[i] - ctr, p[i + 1] - ctr );
    if ( dot( tot, n ) < 0 )
        n = -n;
    V3 xd = p[0] - ctr;
    xd -= n * dot( xd, n );
    if ( !( xd.length() > 0 ) )
        return {};
    xd = xd.normalized();
    const V3 yd = cross( n, xd );
    double prev = 0;
    for ( size_t i = 1; i < p.size(); ++i )
    {
        const V3 d = p[i] - ctr;
        double ang = std::atan2( dot( d, yd ), dot( d, xd ) );
        while ( ang < prev - PI )
            ang += 2 * PI;
        while ( ang > prev + PI )
            ang -= 2 * PI;
        if ( closed && i + 1 == p.size() )
            ang = 2 * PI;
        if ( ang <= prev )
            return {};
        // nothing but the points defines the curve on mesh boundary, and few points can be the corners of a polygon
        if ( ss.size() < 2 && ang - prev > cMaxFacetAngle )
            return {};
        prev = ang;
    }
    if ( prev > 2 * PI + 1e-9 )
        return {};
    std::vector<V3> samples;
    const int num = 4 * (int)p.size();
    for ( int i = 0; i <= num; ++i )
    {
        const double t = prev * i / num;
        samples.push_back( ctr + r * ( std::cos( t ) * xd + std::sin( t ) * yd ) );
    }
    if ( onSurfs_( samples, ss ) > curveTol_( ss ) )
        return {};
    Curve res;
    res.kind = Curve::Kind::Circle;
    res.c = ctr;
    res.n = n;
    res.x = xd;
    res.r = r;
    return res;
}

/// circle around the axis of an adjacent surface of revolution: steadier than a free fit to few points
std::optional<Curve> BRepWriter::axisCircle_( const std::vector<V3>& p, const Surfs& ss, bool closed ) const
{
    for ( auto s : ss )
    {
        if ( s->type != SurfType::Cyl && s->type != SurfType::Cone && s->type != SurfType::Torus )
            continue;
        double minH = DBL_MAX, maxH = -DBL_MAX, minR = DBL_MAX, maxR = -DBL_MAX, sumH = 0, sumR = 0;
        for ( const auto& q : p )
        {
            const V3 d = q - s->c;
            const double h = dot( d, s->a );
            const double rho = ( d - h * s->a ).length();
            minH = std::min( minH, h );
            maxH = std::max( maxH, h );
            minR = std::min( minR, rho );
            maxR = std::max( maxR, rho );
            sumH += h;
            sumR += rho;
        }
        const double rad = sumR / double( p.size() );
        if ( maxH - minH > d_.tol || maxR - minR > d_.tol || rad < 1e-9 * d_.diag )
            continue;
        if ( auto res = circleFrom_( p, s->c + s->a * ( sumH / double( p.size() ) ), s->a, rad, ss, closed ) )
            return res;
    }
    return {};
}

std::optional<Curve> BRepWriter::tryCircle_( const std::vector<V3>& p, const Surfs& ss, bool closed ) const
{
    std::vector<V3> q( p.begin(), closed ? p.end() - 1 : p.end() );
    if ( q.size() < 3 )
        return {};
    const V3 c0 = centroid( q );
    SymMatrix3d cov;
    for ( const auto& pt : q )
        cov += outerSquare( pt - c0 );
    Matrix3d vecs;
    const V3 vals = cov.eigens( &vecs );
    if ( std::sqrt( std::max( vals[1], 0.0 ) ) < 1e-6 * std::sqrt( std::max( vals[2], 1e-24 ) ) )
        return {};
    const V3 n = vecs.x, u = vecs.z, v = vecs.y;
    std::vector<double> xs, ys;
    for ( const auto& pt : q )
    {
        if ( std::abs( dot( pt - c0, n ) ) > 0.5 * d_.tol )
            return {};
        xs.push_back( dot( pt - c0, u ) );
        ys.push_back( dot( pt - c0, v ) );
    }
    double cx, cy, r;
    if ( !fitCircle2( xs, ys, cx, cy, r ) || r < 1e-9 * d_.diag )
        return {};
    const V3 ctr = c0 + cx * u + cy * v;
    for ( const auto& pt : q )
        if ( std::abs( ( pt - ctr ).length() - r ) > 0.5 * d_.tol )
            return {};
    return circleFrom_( p, ctr, n, r, ss, closed );
}

/// B-spline curve interpolating given points and additional points between them projected on the surfaces
Curve BRepWriter::spline_( const std::vector<V3>& p, const Surfs& ss ) const
{
    auto q = p;
    for ( int iter = 0; iter < 2; ++iter )
    {
        std::vector<V3> z;
        z.reserve( 2 * q.size() );
        for ( size_t i = 0; i + 1 < q.size(); ++i )
        {
            z.push_back( q[i] );
            V3 m = ( q[i] + q[i + 1] ) / 2.0;
            for ( int it = 0; it < 30; ++it )
                for ( auto s : ss )
                    m -= s->dist( m ) * s->normal( m );
            z.push_back( m );
        }
        z.push_back( q.back() );
        q = std::move( z );
    }
    const int n = (int)q.size();
    auto project = [&]( V3 m )
    {
        for ( int it = 0; it < 30; ++it )
            for ( auto s : ss )
                m -= s->dist( m ) * s->normal( m );
        return m;
    };
    {
        // evenly spaced points make the interpolation stable
        std::vector<double> len( n, 0.0 );
        for ( int i = 1; i < n; ++i )
            len[i] = len[i - 1] + ( q[i] - q[i - 1] ).length();
        std::vector<V3> even( n );
        even.front() = q.front();
        even.back() = q.back();
        for ( int i = 1, j = 1; i + 1 < n; ++i )
        {
            const double l = len.back() * i / ( n - 1 );
            while ( j + 1 < n && len[j] < l )
                ++j;
            const double seg = len[j] - len[j - 1];
            const double a = seg > 0 ? ( l - len[j - 1] ) / seg : 0;
            even[i] = project( q[j - 1] * ( 1 - a ) + q[j] * a );
        }
        q = std::move( even );
    }
    std::vector<double> t( n, 0.0 );
    for ( int i = 1; i < n; ++i )
        t[i] = t[i - 1] + ( q[i] - q[i - 1] ).length();
    for ( auto& x : t )
        x /= std::max( t.back(), 1e-300 );
    t.back() = 1;

    Curve res;
    res.kind = Curve::Kind::BSpline;
    res.deg = std::min( 3, n - 1 );
    const int k = res.deg;
    res.knots.assign( n + k + 1, 1.0 );
    for ( int i = 0; i <= k; ++i )
        res.knots[i] = 0;
    for ( int j = 1; j < n - k; ++j )
        res.knots[j + k] = std::accumulate( t.begin() + j, t.begin() + j + k, 0.0 ) / k;
    Eigen::MatrixXd m = Eigen::MatrixXd::Zero( n, n );
    Eigen::MatrixXd rhs( n, 3 );
    for ( int i = 0; i < n; ++i )
    {
        double b[4];
        const int i0 = bsplineBasis( res.knots, k, n, t[i], b, nullptr );
        for ( int j = 0; j <= k; ++j )
            m( i, i0 + j ) = b[j];
        for ( int c = 0; c < 3; ++c )
            rhs( i, c ) = q[i][c];
    }
    const Eigen::MatrixXd sol = m.partialPivLu().solve( rhs );
    res.cp.resize( n );
    for ( int i = 0; i < n; ++i )
        res.cp[i] = V3( sol( i, 0 ), sol( i, 1 ), sol( i, 2 ) );

    std::vector<V3> samples;
    const int num = 8 * n;
    for ( int i = 0; i < num; ++i )
    {
        double b[4];
        const int i0 = bsplineBasis( res.knots, k, n, double( i ) / ( num - 1 ), b, nullptr );
        V3 s;
        for ( int j = 0; j <= k; ++j )
            s += b[j] * res.cp[i0 + j];
        samples.push_back( s );
    }
    res.dev = onSurfs_( samples, ss );
    if ( !std::isfinite( res.dev ) )
        res.dev = DBL_MAX;
    return res;
}

/// approximates the chain of mesh vertices by a line or a circle if possible, otherwise splits it in the corners and approximates by splines
void BRepWriter::splitFit_( const std::vector<int>& ids, const Surfs& ss )
{
    const auto p = pointsOf_( ids );
    const bool closed = ids.front() == ids.back();
    std::optional<Curve> c;
    if ( !closed )
        c = tryLine_( p, ss );
    if ( !c )
        c = axisCircle_( p, ss, closed );
    if ( !c )
        c = tryCircle_( p, ss, closed );
    if ( c )
    {
        edges_.push_back( { ids, *c } );
        return;
    }
    static const double cCorner = 22 * PI / 180;
    const int n = (int)ids.size();
    auto sub = [&]( int from, int to ) { return std::vector<int>( ids.begin() + from, ids.begin() + to + 1 ); };
    if ( n > 2 )
    {
        std::vector<V3> dirs;
        for ( int i = 0; i + 1 < n; ++i )
            dirs.push_back( ( p[i + 1] - p[i] ).normalized() );
        auto angle = [&]( int i, int j ) { return std::acos( std::clamp( dot( dirs[i], dirs[j] ), -1.0, 1.0 ) ); };
        if ( closed )
        {
            // angle[0] is in the start vertex, angle[i] is in vertex i
            int best = 0;
            double bestAng = angle( n - 2, 0 );
            for ( int i = 1; i + 1 < n; ++i )
                if ( double a = angle( i - 1, i ); a > bestAng )
                {
                    bestAng = a;
                    best = i;
                }
            if ( bestAng > cCorner && best != 0 )
            {
                // restart the closed chain in its sharpest corner
                std::vector<int> rotated( ids.begin() + best, ids.end() - 1 );
                rotated.insert( rotated.end(), ids.begin(), ids.begin() + best + 1 );
                return splitFit_( rotated, ss );
            }
            if ( bestAng <= cCorner )
            {
                // smooth closed non-circular curve: cut it in two halves
                const int h = n / 2;
                splitFit_( sub( 0, h ), ss );
                splitFit_( sub( h, n - 1 ), ss );
                return;
            }
        }
        std::vector<int> corners;
        for ( int i = 1; i + 1 < n; ++i )
            if ( angle( i - 1, i ) > cCorner )
                corners.push_back( i );
        if ( !corners.empty() )
        {
            corners.push_back( n - 1 );
            int prev = 0;
            for ( int i : corners )
            {
                splitFit_( sub( prev, i ), ss );
                prev = i;
            }
            return;
        }
    }
    Curve sp = spline_( p, ss );
    if ( n == 2 && sp.dev >= curveTol_( ss ) )
        sp = Curve{}; // no better curve than a line segment
    edges_.push_back( { ids, std::move( sp ) } );
}

void BRepWriter::findEdges_()
{
    MR_TIMER;
    const auto& d = d_;
    std::map<int, std::set<int>> adj;
    for ( int f = 0; f < d.nf; ++f )
    {
        if ( !d.valid( f ) )
            continue;
        for ( int i = 0; i < 3; ++i )
        {
            const int g = d.nbr[f][i];
            if ( g >= 0 && d.sub[g] == d.sub[f] )
                continue;
            const int u = d.tris[f][i], v = d.tris[f][( i + 1 ) % 3];
            bnd_[{ u, v }] = { f, i };
            adj[u].insert( v );
            adj[v].insert( u );
        }
    }
    auto isNode = [&]( int v ) { return adj[v].size() != 2; };
    std::set<DirEdge> visited;
    auto walk = [&]( int u, int v )
    {
        std::vector<int> vs{ u, v };
        while ( !isNode( vs.back() ) && vs.back() != vs.front() )
        {
            const auto& s = adj[vs.back()];
            const int prev = vs[vs.size() - 2];
            vs.push_back( *s.begin() == prev ? *s.rbegin() : *s.begin() );
        }
        for ( size_t i = 0; i + 1 < vs.size(); ++i )
        {
            visited.insert( { vs[i], vs[i + 1] } );
            visited.insert( { vs[i + 1], vs[i] } );
        }
        // the surfaces on both sides of the chain
        Surfs ss;
        for ( const DirEdge& de : { DirEdge{ u, v }, DirEdge{ v, u } } )
            if ( auto it = bnd_.find( de ); it != bnd_.end() )
                ss.push_back( &d.regions[d.sub[it->second.first]].surf );
        const size_t first = edges_.size();
        splitFit_( vs, ss );
        for ( size_t e = first; e < edges_.size(); ++e )
        {
            const auto& ids = edges_[e].verts;
            for ( size_t i = 0; i + 1 < ids.size(); ++i )
            {
                edgeOf_[{ ids[i], ids[i + 1] }] = { (int)e, true, i == 0 };
                edgeOf_[{ ids[i + 1], ids[i] }] = { (int)e, false, i + 2 == ids.size() };
            }
        }
    };
    for ( const auto& [u, s] : adj )
        if ( s.size() != 2 )
            for ( int v : std::vector<int>( s.begin(), s.end() ) )
                if ( !visited.count( { u, v } ) )
                    walk( u, v );
    // closed chains without nodes
    for ( const auto& [de, fi] : std::map<DirEdge, std::pair<int, int>>( bnd_ ) )
        if ( !visited.count( de ) )
            walk( de.first, de.second );
}

void BRepWriter::findLoops_()
{
    const auto& d = d_;
    loops_.resize( d.regions.size() );
    std::set<DirEdge> done;
    for ( const auto& [start, fi] : bnd_ )
    {
        if ( done.count( start ) )
            continue;
        const int r = d.sub[fi.first];
        std::vector<std::pair<int, bool>> loop;
        for ( DirEdge cur = start; done.insert( cur ).second; )
        {
            const auto& use = edgeOf_.at( cur );
            if ( use.first )
                loop.push_back( { use.edge, use.forward } );
            // the next boundary edge of the same region: rotate around the destination vertex
            int f = bnd_.at( cur ).first;
            for ( ;; )
            {
                const int i = int( std::find( d.tris[f].begin(), d.tris[f].end(), cur.second ) - d.tris[f].begin() );
                const int g = d.nbr[f][i];
                if ( g < 0 || d.sub[g] != r )
                {
                    cur = { cur.second, d.tris[f][( i + 1 ) % 3] };
                    break;
                }
                f = g;
            }
        }
        loops_[r].push_back( std::move( loop ) );
    }
}

std::string BRepWriter::add_( std::string s )
{
    lines_.push_back( std::move( s ) );
    return fmt::format( "#{}", lines_.size() );
}

/// real number in the form required by STEP: with a decimal point
std::string stepReal( double x )
{
    if ( !std::isfinite( x ) )
        x = 0;
    std::string s = fmt::format( "{}", x );
    const auto e = s.find( 'e' );
    if ( e == std::string::npos )
    {
        if ( s.find( '.' ) == std::string::npos )
            s += '.';
        return s;
    }
    std::string mant = s.substr( 0, e );
    if ( mant.find( '.' ) == std::string::npos )
        mant += '.';
    return mant + "E" + std::to_string( std::stoi( s.substr( e + 1 ) ) );
}

double rounded( double x, double mult )
{
    return std::round( x * mult ) / mult + 0.0;
}

std::string BRepWriter::point_( const V3& p )
{
    return add_( fmt::format( "CARTESIAN_POINT('',({},{},{}))", stepReal( rounded( p.x, 1e9 ) ), stepReal( rounded( p.y, 1e9 ) ), stepReal( rounded( p.z, 1e9 ) ) ) );
}

std::string BRepWriter::dir_( const V3& d0 )
{
    const V3 d = d0.normalized();
    return add_( fmt::format( "DIRECTION('',({},{},{}))", stepReal( rounded( d.x, 1e12 ) ), stepReal( rounded( d.y, 1e12 ) ), stepReal( rounded( d.z, 1e12 ) ) ) );
}

std::string BRepWriter::axis2_( const V3& p, const V3& z0, const V3* x0 )
{
    const V3 z = z0.normalized();
    V3 x;
    if ( x0 )
        x = *x0;
    else
    {
        // the reference direction sets the seam of a periodic surface, and the reader leaves small cracks
        // near some seams; this choice (not the one of Vector3d::perpendicular) gave the fewest of them
        x = V3( 1, 0, 0 );
        if ( std::abs( z.y ) < std::abs( z.x ) && std::abs( z.y ) <= std::abs( z.z ) )
            x = V3( 0, 1, 0 );
        else if ( std::abs( z.z ) < std::abs( z.x ) && std::abs( z.z ) < std::abs( z.y ) )
            x = V3( 0, 0, 1 );
        x = cross( z, x );
    }
    x -= z * dot( x, z );
    const auto ps = point_( p );
    const auto zs = dir_( z );
    return add_( fmt::format( "AXIS2_PLACEMENT_3D('',{},{},{})", ps, zs, dir_( x ) ) );
}

std::string BRepWriter::vertex_( int v )
{
    auto [it, inserted] = vertexIds_.insert( { v, {} } );
    if ( inserted )
        it->second = add_( fmt::format( "VERTEX_POINT('',{})", point_( d_.pts[v] ) ) );
    return it->second;
}

/// makes the direction numerically parallel to a coordinate axis exactly parallel
V3 snapped( const V3& a )
{
    for ( int i = 0; i < 3; ++i )
        if ( std::abs( std::abs( a[i] ) - 1 ) < 1e-9 )
        {
            V3 res;
            res[i] = a[i] > 0 ? 1 : -1;
            return res;
        }
    return a;
}

std::string joined( const std::vector<std::string>& v )
{
    std::string res;
    for ( const auto& s : v )
    {
        if ( !res.empty() )
            res += ',';
        res += s;
    }
    return res;
}

/// distinct knots and their multiplicities as two STEP lists
std::pair<std::string, std::string> knotLists( const std::vector<double>& knots )
{
    std::vector<std::string> ks, ms;
    for ( size_t i = 0; i < knots.size(); )
    {
        size_t j = i;
        while ( j < knots.size() && knots[j] == knots[i] )
            ++j;
        ks.push_back( stepReal( knots[i] ) );
        ms.push_back( std::to_string( j - i ) );
        i = j;
    }
    return { joined( ms ), joined( ks ) };
}

MeshToStepStats BRepWriter::write( std::ostream& out )
{
    MR_TIMER;
    const auto& d = d_;
    findEdges_();
    findLoops_();

    const auto app = add_( "APPLICATION_CONTEXT('automotive design')" );
    add_( fmt::format( "APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2010,{})", app ) );
    const auto pc = add_( fmt::format( "PRODUCT_CONTEXT('',{},'mechanical')", app ) );
    const auto prod = add_( fmt::format( "PRODUCT('{0}','{0}','',({1}))", name_, pc ) );
    const auto pdf = add_( fmt::format( "PRODUCT_DEFINITION_FORMATION('','',{})", prod ) );
    const auto pdc = add_( fmt::format( "PRODUCT_DEFINITION_CONTEXT('part definition',{},'design')", app ) );
    const auto pd = add_( fmt::format( "PRODUCT_DEFINITION('design','',{},{})", pdf, pdc ) );
    const auto pds = add_( fmt::format( "PRODUCT_DEFINITION_SHAPE('','',{})", pd ) );
    add_( fmt::format( "PRODUCT_RELATED_PRODUCT_CATEGORY('part','',({}))", prod ) );
    const auto lu = add_( "(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))" );
    const auto au = add_( "(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))" );
    const auto su = add_( "(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())" );
    const auto unc = add_( fmt::format( "UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE({}),{},'distance_accuracy_value','')", stepReal( d.tol ), lu ) );
    const auto ctx = add_( fmt::format( "(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT(({}))GLOBAL_UNIT_ASSIGNED_CONTEXT(({},{},{}))REPRESENTATION_CONTEXT('',''))", unc, lu, au, su ) );

    std::vector<std::string> edgeIds;
    for ( const auto& e : edges_ )
    {
        const auto& cv = e.curve;
        const V3& p0 = d.pts[e.verts.front()];
        const V3& p1 = d.pts[e.verts.back()];
        std::string geom;
        if ( cv.kind == Curve::Kind::Line )
        {
            const auto ps = point_( p0 );
            const auto vec = add_( fmt::format( "VECTOR('',{},{})", dir_( p1 - p0 ), stepReal( ( p1 - p0 ).length() ) ) );
            geom = add_( fmt::format( "LINE('',{},{})", ps, vec ) );
        }
        else if ( cv.kind == Curve::Kind::Circle )
            geom = add_( fmt::format( "CIRCLE('',{},{})", axis2_( cv.c, snapped( cv.n ), &cv.x ), stepReal( rounded( cv.r, 1e9 ) ) ) );
        else
        {
            std::vector<std::string> cps;
            for ( const auto& p : cv.cp )
                cps.push_back( point_( p ) );
            const auto [mults, knots] = knotLists( cv.knots );
            geom = add_( fmt::format( "B_SPLINE_CURVE_WITH_KNOTS('',{},({}),.UNSPECIFIED.,.F.,.F.,({}),({}),.UNSPECIFIED.)", cv.deg, joined( cps ), mults, knots ) );
        }
        const auto v0 = vertex_( e.verts.front() );
        const auto v1 = vertex_( e.verts.back() );
        edgeIds.push_back( add_( fmt::format( "EDGE_CURVE('',{},{},{},.T.)", v0, v1, geom ) ) );
    }

    MeshToStepStats stats;
    const auto regFaces = d.facesOfRegions();
    std::vector<std::string> faceIds( d.regions.size() );
    for ( size_t r = 0; r < d.regions.size(); ++r )
    {
        const Surf& s = d.regions[r].surf;
        const auto& fs = regFaces[r];
        if ( fs.empty() )
            continue;
        const auto verts = d.vertsOf( fs );
        double minH = DBL_MAX, maxH = -DBL_MAX;
        for ( int v : verts )
        {
            const double h = dot( d.pts[v] - s.c, s.a );
            minH = std::min( minH, h );
            maxH = std::max( maxH, h );
        }
        std::string srf;
        switch ( s.type )
        {
        case SurfType::Plane:
            ++stats.planes;
            srf = add_( fmt::format( "PLANE('',{})", axis2_( s.c, snapped( s.a ) ) ) );
            break;
        case SurfType::Cyl:
            ++stats.cylinders;
            srf = add_( fmt::format( "CYLINDRICAL_SURFACE('',{},{})", axis2_( s.c + s.a * minH, snapped( s.a ) ), stepReal( rounded( s.r, 1e9 ) ) ) );
            break;
        case SurfType::Cone:
        {
            ++stats.cones;
            const double h0 = ( minH + maxH ) / 2;
            srf = add_( fmt::format( "CONICAL_SURFACE('',{},{},{})", axis2_( s.c + s.a * h0, snapped( s.a ) ), stepReal( std::abs( h0 ) * std::tan( s.r ) ), stepReal( s.r ) ) );
            break;
        }
        case SurfType::Sphere:
            ++stats.spheres;
            srf = add_( fmt::format( "SPHERICAL_SURFACE('',{},{})", axis2_( s.c, V3( 0, 0, 1 ) ), stepReal( rounded( s.r, 1e9 ) ) ) );
            break;
        case SurfType::Torus:
            ++stats.tori;
            srf = add_( fmt::format( "TOROIDAL_SURFACE('',{},{},{})", axis2_( s.c, snapped( s.a ) ), stepReal( rounded( s.R, 1e9 ) ), stepReal( rounded( s.r, 1e9 ) ) ) );
            break;
        case SurfType::BSpline:
        {
            ++stats.bsplines;
            stats.maxBSplineError = std::max( stats.maxBSplineError, float( s.bs->err ) );
            std::vector<std::string> rows;
            for ( int i = 0; i < s.bs->nu; ++i )
            {
                std::vector<std::string> row;
                for ( int j = 0; j < s.bs->nv; ++j )
                    row.push_back( point_( s.bs->cp[i * s.bs->nv + j] ) );
                rows.push_back( fmt::format( "({})", joined( row ) ) );
            }
            const auto [mu, ku] = knotLists( s.bs->tu );
            const auto [mv, kv] = knotLists( s.bs->tv );
            srf = add_( fmt::format( "B_SPLINE_SURFACE_WITH_KNOTS('',{0},{0},({1}),.UNSPECIFIED.,.F.,.F.,.F.,({2}),({3}),({4}),({5}),.UNSPECIFIED.)",
                BSplineSurf::deg, joined( rows ), mu, mv, ku, kv ) );
            break;
        }
        }
        double sense = 0;
        for ( int f : fs )
            sense += dot( s.normal( d.centers[f] ), d.normals[f] ) * d.areas[f];

        // the loop with the largest bounding box is the outer one
        const auto& loops = loops_[r];
        std::vector<double> extent;
        std::set<int> loopVerts;
        for ( const auto& loop : loops )
        {
            Box3d box;
            for ( const auto& [e, fwd] : loop )
                for ( int v : edges_[e].verts )
                {
                    box.include( d.pts[v] );
                    loopVerts.insert( v );
                }
            extent.push_back( box.valid() ? box.diagonal() : 0 );
        }
        std::vector<int> order( loops.size() );
        std::iota( order.begin(), order.end(), 0 );
        std::stable_sort( order.begin(), order.end(), [&]( int x, int y ) { return extent[x] > extent[y]; } );
        std::vector<std::string> bounds;
        for ( int li : order )
        {
            std::vector<std::string> oes;
            for ( const auto& [e, fwd] : loops[li] )
                oes.push_back( add_( fmt::format( "ORIENTED_EDGE('',*,*,{},{})", edgeIds[e], fwd ? ".T." : ".F." ) ) );
            const auto el = add_( fmt::format( "EDGE_LOOP('',({}))", joined( oes ) ) );
            bounds.push_back( add_( fmt::format( "{}('',{},.T.)", bounds.empty() && s.type == SurfType::Plane ? "FACE_OUTER_BOUND" : "FACE_BOUND", el ) ) );
        }
        auto addVertexLoop = [&]( int v )
        {
            const auto vl = add_( fmt::format( "VERTEX_LOOP('',{})", vertex_( v ) ) );
            bounds.push_back( add_( fmt::format( "FACE_BOUND('',{},.T.)", vl ) ) );
        };
        if ( s.type == SurfType::Cone )
        {
            // the apex inside the face
            int best = verts[0];
            for ( int v : verts )
                if ( ( d.pts[v] - s.c ).lengthSq() < ( d.pts[best] - s.c ).lengthSq() )
                    best = v;
            if ( ( d.pts[best] - s.c ).length() < 5 * d.tol && !loopVerts.count( best ) )
                addVertexLoop( best );
        }
        if ( bounds.empty() )
            addVertexLoop( verts[0] );
        faceIds[r] = add_( fmt::format( "ADVANCED_FACE('',({}),{},{})", joined( bounds ), srf, sense > 0 ? ".T." : ".F." ) );
    }

    // every connected component of the mesh makes its own shell
    std::vector<int> comp( d.nf, -1 );
    std::vector<std::string> shells;
    bool allClosed = true;
    std::vector<int> stack;
    for ( int f0 = 0; f0 < d.nf; ++f0 )
    {
        if ( !d.valid( f0 ) || comp[f0] >= 0 )
            continue;
        const int id = (int)shells.size();
        std::set<int> regs;
        bool closed = true;
        comp[f0] = id;
        stack.push_back( f0 );
        while ( !stack.empty() )
        {
            const int f = stack.back();
            stack.pop_back();
            regs.insert( d.sub[f] );
            for ( int g : d.nbr[f] )
            {
                if ( g < 0 )
                    closed = false;
                else if ( comp[g] < 0 )
                {
                    comp[g] = id;
                    stack.push_back( g );
                }
            }
        }
        std::vector<std::string> fids;
        for ( int r : regs )
            fids.push_back( faceIds[r] );
        shells.push_back( add_( fmt::format( "{}('',({}))", closed ? "CLOSED_SHELL" : "OPEN_SHELL", joined( fids ) ) ) );
        allClosed = allClosed && closed;
    }
    const V3 xAxis( 1, 0, 0 );
    const auto origin = axis2_( V3(), V3( 0, 0, 1 ), &xAxis );
    std::string rep;
    if ( allClosed )
    {
        std::vector<std::string> items{ origin };
        for ( const auto& sh : shells )
            items.push_back( add_( fmt::format( "MANIFOLD_SOLID_BREP('{}',{})", name_, sh ) ) );
        rep = add_( fmt::format( "ADVANCED_BREP_SHAPE_REPRESENTATION('',({}),{})", joined( items ), ctx ) );
    }
    else
    {
        const auto model = add_( fmt::format( "SHELL_BASED_SURFACE_MODEL('{}',({}))", name_, joined( shells ) ) );
        rep = add_( fmt::format( "MANIFOLD_SURFACE_SHAPE_REPRESENTATION('',({},{}),{})", origin, model, ctx ) );
    }
    add_( fmt::format( "SHAPE_DEFINITION_REPRESENTATION({},{})", pds, rep ) );

    out << "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('surfaces fitted to a mesh'),'2;1');\n"
        << "FILE_NAME('" << name_ << "','',(''),(''),'MeshLib','MeshLib','');\n"
        << "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 3 1 1 }'));\nENDSEC;\nDATA;\n";
    for ( size_t i = 0; i < lines_.size(); ++i )
        out << '#' << i + 1 << '=' << lines_[i] << ";\n";
    out << "ENDSEC;\nEND-ISO-10303-21;\n";
    return stats;
}

} //anonymous namespace

MeshToStepStats writeBRep( const Data& d, const std::string& name, std::ostream& out )
{
    return BRepWriter( d, name ).write( out );
}

} //namespace MeshToStep

} //namespace MR
