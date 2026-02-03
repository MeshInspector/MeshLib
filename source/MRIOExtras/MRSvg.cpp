#include "MRSvg.h"
#ifndef MRIOEXTRAS_NO_XML

#include "MRMesh/MRBezier.h"
#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRIOParsing.h"
#include "MRMesh/MRMatrix2.h"
#include "MRMesh/MRStringConvert.h"

#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/spirit/home/x3.hpp>

#include <tinyxml2.h>

#include <variant>

// required for parsing lists of points
BOOST_FUSION_ADAPT_STRUCT( MR::Vector2f, x, y )

namespace MR::LinesLoad
{

namespace
{

using namespace tinyxml2;

namespace Path
{

struct MoveTo
{
    bool relative = false;
    Vector2f to;
};

struct ClosePath
{
    // this flag has no effect for ClosePath but allows to set it uniformly for other commands
    bool relative = false;
};

struct LineTo
{
    bool relative = false;
    enum Kind
    {
        Tangent,
        Horizontal,
        Vertical,
    } kind = Tangent;
    Vector2f to;
};

struct CubicBezier
{
    bool relative = false;
    bool shorthand = false;
    std::array<Vector2f, 2> controlPoints;
    Vector2f end;
};

struct QuadraticBezier
{
    bool relative = false;
    bool shorthand = false;
    Vector2f controlPoint;
    Vector2f end;
};

struct EllipticalArc
{
    bool relative = false;
    Vector2f radii;
    float xAxisRot = 0.f;
    bool largeArc = false;
    bool sweep = false;
    Vector2f end;
};

using Command = std::variant<MoveTo, ClosePath, LineTo, CubicBezier, QuadraticBezier, EllipticalArc>;

}

namespace parser
{

using namespace boost::spirit::x3;

constexpr auto point = rule<class point, MR::Vector2f>{ "point" }
                     = double_ >> -lit( ',' ) >> double_;
constexpr auto points = point % -lit( ',' );

Expected<std::vector<Vector2f>> parsePoints( std::string_view str )
{
    std::vector<Vector2f> results;
    if ( !phrase_parse( str.begin(), str.end(), points, space, results ) )
        return unexpected( "Failed to parse points" );
    return results;
}

Expected<AffineXf2f> parseTransform( std::string_view str )
{
#define _(...) ( [] ( auto& ctx ) { [[maybe_unused]] auto& xf = _val( ctx ); [[maybe_unused]] auto& A = xf.A; [[maybe_unused]] auto& b = xf.b; [[maybe_unused]] auto& val = _attr( ctx ); __VA_ARGS__; } )

    constexpr auto matrix
        = rule<class matrix, AffineXf2f>{ "matrix" }
        = lit( "matrix" ) >> '('
            >> double_[_( A.x.x = val )] >> -lit( ',' )
            >> double_[_( A.y.x = val )] >> -lit( ',' )
            >> double_[_( A.x.y = val )] >> -lit( ',' )
            >> double_[_( A.y.y = val )] >> -lit( ',' )
            >> double_[_( b.x = val )] >> -lit( ',' )
            >> double_[_( b.y = val )]
          >> ')';
    constexpr auto translate
        = rule<class translate, AffineXf2f>{ "translate" }
        = lit( "translate" ) >> '('
            >> double_[_( b.x = val )] >> -lit( ',' )
            >> -double_[_( b.y = val )]
          >> ')';
    constexpr auto scale
        = rule<class scale, AffineXf2f>{ "scale" }
        = lit( "scale" ) >> '('
            // uniform scaling by default
            >> double_[_( A.x.x = val, A.y.y = val )] >> -lit( ',' )
            >> -double_[_( A.y.y = val )]
          >> ')';
    constexpr auto rotate
        = rule<class rotate, AffineXf2f>{ "rotate" }
        = lit( "rotate" ) >> '('
            >> double_[_( A = Matrix2f::rotation( val * PI_F / 180.f ) )] >> -lit( ',' )
            // optional translation
            >> -( point[_( xf = AffineXf2f::translation( val ) * xf * AffineXf2f::translation( -val ) )])
          >> ')';
    constexpr auto skewX
        = rule<class skewX, AffineXf2f>{ "skewX" }
        = lit( "skewX" ) >> '('
            >> double_[_( A.x.y = std::tan( val * PI_F / 180.f ) )]
          >> ')';
    constexpr auto skewY
        = rule<class skewY, AffineXf2f>{ "skewY" }
        = lit( "skewY" ) >> '('
            >> double_[_( A.y.x = std::tan( val * PI_F / 180.f ) )]
          >> ')';

#undef _

    constexpr auto transform = ( matrix | translate | scale | rotate | skewX | skewY ) % -lit( ',' );

    std::vector<AffineXf2f> results;
    if ( !phrase_parse( str.begin(), str.end(), transform, space, results ) )
        return unexpected( "Failed to parse points" );

    AffineXf2f result;
    for ( auto xf : results )
        result = result * xf;

    return result;
}

Expected<std::vector<Path::Command>> parsePath( std::string_view str )
{
#define _( ... ) ( [] ( auto& ctx ) { auto& val = _val( ctx ); [[maybe_unused]] auto& attr = _attr( ctx ); __VA_ARGS__; } )

    constexpr auto closepath = rule<class closepath, Path::ClosePath>{ "closepath" }
                             = lit( 'Z' ) | lit( 'z' );

    constexpr auto moveto = rule<class moveto, Path::MoveTo>{ "moveto" }
                          = point[_( val.to = attr )];

    constexpr auto lineto = rule<class lineto, Path::LineTo>{ "lineto" }
                          = point[_( val.to = attr )];

    constexpr auto hlineto = rule<class hlineto, Path::LineTo>{ "hlineto" }
                           = eps[_( val.kind = Path::LineTo::Horizontal )] >> double_[_( val.to.x = attr )];

    constexpr auto vlineto = rule<class vlineto, Path::LineTo>{ "vlineto" }
                           = eps[_( val.kind = Path::LineTo::Vertical )] >> double_[_( val.to.y = attr )];

    constexpr auto curveto = rule<class curveto, Path::CubicBezier>{ "curveto" }
                           = point[_( val.controlPoints[0] = attr )] >> point[_( val.controlPoints[1] = attr )] >> point[_( val.end = attr )];

    constexpr auto scurveto = rule<class scurveto, Path::CubicBezier>{ "scurveto" }
                            = eps[_( val.shorthand = true )] >> point[_( val.controlPoints[1] = attr )] >> point[_( val.end = attr )];

    constexpr auto qcurveto = rule<class qcurveto, Path::QuadraticBezier>{ "qcurveto" }
                            = point[_( val.controlPoint = attr )] >> point[_( val.end = attr )];

    constexpr auto qscurveto = rule<class qscurveto, Path::QuadraticBezier>{ "qscurveto" }
                             = eps[_( val.shorthand = true )] >> point[_( val.end = attr )];

    constexpr auto arc = rule<class arc, Path::EllipticalArc>{ "arc" }
                       = point[_( val.radii = attr )] >> double_[_( val.xAxisRot = attr )] >> int_[_( val.largeArc = (bool)attr )] >> int_[_( val.sweep = (bool)attr )] >> point[_( val.end = attr )];

#undef _

    bool relative = false;
    std::vector<Path::Command> commands;
#define ABS ( [&relative] ( auto& ) { relative = false; } )
#define REL ( [&relative] ( auto& ) { relative = true; } )
#define ADD ( [&relative, &commands] ( auto& ctx ) { _attr( ctx ).relative = relative; commands.emplace_back( _attr( ctx ) ); } )

    const auto command = +(
        closepath[ADD]
        // "If a moveto is followed by multiple pairs of coordinates, the subsequent pairs are treated as implicit lineto commands."
        | ( ( lit( 'M' )[ABS] | lit( 'm' )[REL] ) >> moveto[ADD] >> *( lineto[ADD] ) )
        | ( ( lit( 'L' )[ABS] | lit( 'l' )[REL] ) >> +( lineto[ADD] ) )
        | ( ( lit( 'H' )[ABS] | lit( 'h' )[REL] ) >> +( hlineto[ADD] ) )
        | ( ( lit( 'V' )[ABS] | lit( 'v' )[REL] ) >> +( vlineto[ADD] ) )
        | ( ( lit( 'C' )[ABS] | lit( 'c' )[REL] ) >> +( curveto[ADD] ) )
        | ( ( lit( 'S' )[ABS] | lit( 's' )[REL] ) >> +( scurveto[ADD] ) )
        | ( ( lit( 'Q' )[ABS] | lit( 'q' )[REL] ) >> +( qcurveto[ADD] ) )
        | ( ( lit( 'T' )[ABS] | lit( 't' )[REL] ) >> +( qscurveto[ADD] ) )
        | ( ( lit( 'A' )[ABS] | lit( 'a' )[REL] ) >> +( arc[ADD] ) )
    );

#undef REL
#undef ABS

    if ( !phrase_parse( str.begin(), str.end(), command, space | lit( ',' ) ) )
        return unexpected( "Failed to parse path" );
    return commands;
}

} // namespace parser

struct EllipseParams
{
    float cx = 0.f;
    float cy = 0.f;
    float rx = 1.f;
    float ry = 1.f;
    float a0 = 0.f;
    float a1 = 2.f * PI;
    int resolution = 32;
};

Contour2f getEllipsePoints( const EllipseParams params = {} )
{
    Contour2f results;
    for ( auto i = 0; i <= params.resolution; ++i )
    {
        const auto a = params.a0 + ( params.a1 - params.a0 ) * (float)i / (float)params.resolution;
        const auto x = std::cos( a ), y = std::sin( a );
        results.push_back( {
            x * params.rx + params.cx,
            y * params.ry + params.cy,
        } );
    }
    return results;
}

void close( Contour2f& contour )
{
    if ( contour.back() != contour.front() )
        contour.emplace_back( contour.front() );
}

class SvgLoader
{
public:
    explicit SvgLoader( const std::filesystem::path& path )
    {
        const auto pathStr = utf8string( path );
        doc_.LoadFile( pathStr.c_str() );
    }
    explicit SvgLoader( const char* data, size_t size )
    {
        doc_.Parse( data, size );
    }

    explicit operator bool () const { return !doc_.Error(); }
    std::string error() const { return doc_.ErrorStr(); }

    Expected<Polyline2> parse()
    {
        auto* svg = doc_.RootElement();
        if ( !svg || svg->Name() != std::string_view{ "svg" } )
            return unexpected( "Not an SVG document" );

        if ( auto contours = parseChildren_( svg ) )
        {
            Polyline2 result{ *contours };
            // flip Y axis
            for ( auto& p : result.points )
                p.y *= -1.f;
            return result;
        }
        else
        {
            return unexpected( std::move( contours.error() ) );
        }
    }

private:
    Expected<Contours2f> parseChildren_( XMLElement* group ) const
    {
        Contours2f results;
        for ( auto* child = group->FirstChildElement(); child != nullptr; child = child->NextSiblingElement() )
        {
            auto contours = parseElement_( child );
            if ( !contours )
                return contours;

            // https://developer.mozilla.org/en-US/docs/Web/SVG/Reference/Attribute/transform
            if ( const auto* transform = child->FindAttribute( "transform" ) )
            {
                const auto xf = parser::parseTransform( transform->Value() );
                if ( !xf )
                    return unexpected( std::move( xf.error() ) );

                for ( auto& contour : *contours )
                    for ( auto& p : contour )
                        p = (*xf)( p );
            }

            for ( auto&& contour : *contours )
                results.emplace_back( std::move( contour ) );
        }
        return results;
    }

    Expected<Contours2f> parseElement_( XMLElement* elem ) const
    {
        const std::string_view name = elem->Name();

        // group
        if ( name == "g" )
            return parseChildren_( elem );
        // path
        if ( name == "path" )
            return parsePath_( elem );
        // basic shapes
        if ( name == "circle" )
            return parseCircle_( elem );
        if ( name == "ellipse" )
            return parseEllipse_( elem );
        if ( name == "line" )
            return parseLine_( elem );
        if ( name == "polygon" )
            return parsePolygon_( elem );
        if ( name == "polyline" )
            return parsePolyline_( elem );
        if ( name == "rect" )
            return parseRect_( elem );

        // omitting unknown element
        return {};
    }

    Expected<Contours2f> parsePath_( XMLElement* elem ) const
    {
        const std::string_view d = elem->Attribute( "d" );
        const auto commands = parser::parsePath( d );
        if ( !commands )
            return unexpected( std::move( commands.error() ) );

        Contours2f results;
        Vector2f pos{};
        Vector2f prevCurveControlPoint{};
        results.emplace_back();
        for ( const auto& command : *commands )
        {
            std::visit( overloaded {
                [&] ( Path::ClosePath )
                {
                    close( results.back() );
                    results.emplace_back();
                },
                [&] ( const Path::MoveTo& cmd )
                {
                    if ( !results.back().empty() )
                        results.emplace_back();

                    if ( cmd.relative )
                        pos += cmd.to;
                    else
                        pos = cmd.to;

                    results.back().emplace_back( pos );
                },
                [&] ( const Path::LineTo& cmd )
                {
                    if ( results.back().empty() )
                        results.back().emplace_back( pos );

                    if ( cmd.relative )
                    {
                        pos += cmd.to;
                    }
                    else
                    {
                        switch ( cmd.kind )
                        {
                        case Path::LineTo::Tangent:
                            pos = cmd.to;
                            break;
                        case Path::LineTo::Horizontal:
                            pos.x = cmd.to.x;
                            break;
                        case Path::LineTo::Vertical:
                            pos.y = cmd.to.y;
                            break;
                        }
                    }

                    results.back().emplace_back( pos );
                },
                [&] ( const Path::CubicBezier& cmd )
                {
                    if ( results.back().empty() )
                        results.back().emplace_back( pos );

                    CubicBezierCurve2f curve;
                    curve.p[0] = pos;
                    if ( cmd.relative )
                    {
                        curve.p[1] = cmd.shorthand ? pos - ( prevCurveControlPoint - pos ) : pos + cmd.controlPoints[0];
                        curve.p[2] = pos + cmd.controlPoints[1];
                        curve.p[3] = pos + cmd.end;
                    }
                    else
                    {
                        curve.p[1] = cmd.shorthand ? pos - ( prevCurveControlPoint - pos ) : cmd.controlPoints[0];
                        curve.p[2] = cmd.controlPoints[1];
                        curve.p[3] = cmd.end;
                    }
                    constexpr auto cResolution = 32;
                    for ( auto i = 1; i <= cResolution; ++i )
                    {
                        const auto t = (float)i / (float)cResolution;
                        results.back().emplace_back( curve.getPoint( t ) );
                    }

                    if ( cmd.relative )
                        pos += cmd.end;
                    else
                        pos = cmd.end;
                    prevCurveControlPoint = cmd.controlPoints[1];
                },
                [&] ( const Path::QuadraticBezier& cmd )
                {
                    if ( results.back().empty() )
                        results.back().emplace_back( pos );

                    std::array<Vector2f, 3> p;
                    p[0] = pos;
                    if ( cmd.relative )
                    {
                        p[1] = cmd.shorthand ? pos - ( prevCurveControlPoint - pos ) : pos + cmd.controlPoint;
                        p[2] = pos + cmd.end;
                    }
                    else
                    {
                        p[1] = cmd.shorthand ? pos - ( prevCurveControlPoint - pos ) : cmd.controlPoint;
                        p[2] = cmd.end;
                    }
                    constexpr auto cResolution = 32;
                    for ( auto i = 1; i <= cResolution; ++i )
                    {
                        const auto t = (float)i / (float)cResolution;
                        const auto q0 = lerp( p[0], p[1], t );
                        const auto q1 = lerp( p[1], p[2], t );
                        const auto r = lerp( q0, q1, t );
                        results.back().emplace_back( r );
                    }

                    if ( cmd.relative )
                        pos += cmd.end;
                    else
                        pos = cmd.end;
                    prevCurveControlPoint = cmd.controlPoint;
                },
                [&] ( const Path::EllipticalArc& cmd )
                {
                    if ( results.back().empty() )
                        results.back().emplace_back( pos );

                    const auto p1 = pos;
                    const auto p2 = cmd.relative ? pos + cmd.end : cmd.end;
                    const auto phi = cmd.xAxisRot * PI_F / 180.f;

                    const auto rot = Matrix2f::rotation( phi );
                    const auto p0_ = rot.transposed() * ( p1 - p2 ) / 2.f;

                    // https://www.w3.org/TR/SVG2/implnote.html#ArcCorrectionOutOfRangeRadii
                    auto r = cmd.radii;
                    if ( r == Vector2f{} )
                        return (void)results.back().emplace_back( p2 );
                    if ( r.x < 0.f )
                        r.x *= -1.f;
                    if ( r.y < 0.f )
                        r.y *= -1.f;
                    const auto lam = div( p0_, r ).lengthSq();
                    if ( lam > 1.f )
                        r = r * std::sqrt( lam );

                    // https://www.w3.org/TR/SVG/implnote.html#ArcConversionEndpointToCenter
                    const auto rp_ = Vector2f { r.x * p0_.y, r.y * p0_.x };
                    assert( rp_.lengthSq() != 0.f );
                    const auto k1Sq = sqr( r.x * r.y ) / rp_.lengthSq() - 1;
                    assert( k1Sq >= 0.f );
                    const auto k1 = std::sqrt( k1Sq ) * ( cmd.largeArc != cmd.sweep ? +1.f : -1.f );
                    assert( std::isfinite( k1 ) );
                    const auto c_ = k1 * Vector2f { rp_.x / r.y, -rp_.y / r.x };
                    const auto c = rot * c_ + ( p1 + p2 ) / 2.f;
                    const auto th1 = angle( Vector2f::plusX(), div( p0_ - c_, r ) );
                    auto dth = angle( div( p0_ - c_, r ), div( -p0_ - c_, r ) );
                    if ( cmd.sweep && dth < 0.f )
                        dth += 2.f * PI_F;
                    if ( !cmd.sweep && dth > 0.f )
                        dth -= 2.f * PI_F;

                    auto arcPoints = getEllipsePoints( {
                        .cx = c.x,
                        .cy = c.y,
                        .rx = r.x,
                        .ry = r.y,
                        .a0 = th1,
                        .a1 = th1 + dth,
                    } );
                    const auto ellXfInv = AffineXf2f::xfAround( rot, c );
                    for ( auto& p : arcPoints )
                        p = ellXfInv( p );
                    assert( ( p1 - arcPoints.front() ).lengthSq() < 1e-6f );
                    assert( ( p2 - arcPoints.back() ).lengthSq() < 1e-6f );
                    for ( auto p : arcPoints )
                        results.back().emplace_back( p );

                    if ( cmd.relative )
                        pos += cmd.end;
                    else
                        pos = cmd.end;
                },
            }, command );
        }
        // remove degenerate contours
        std::erase_if( results, [] ( auto&& contour )
        {
            return contour.size() <= 1;
        } );
        return results;
    }

    Expected<Contours2f> parseCircle_( XMLElement* elem ) const
    {
        const auto cx = elem->FloatAttribute( "cx", 0.f );
        const auto cy = elem->FloatAttribute( "cy", 0.f );
        const auto r = elem->FloatAttribute( "r", 0.f );
        if ( r == 0.f )
            return {};
        auto points = getEllipsePoints( {
            .cx = cx,
            .cy = cy,
            .rx = r,
            .ry = r,
        } );
        return { { std::move( points ) } };
    }

    Expected<Contours2f> parseEllipse_( XMLElement* elem ) const
    {
        const auto cx = elem->FloatAttribute( "cx", 0.f );
        const auto cy = elem->FloatAttribute( "cy", 0.f );
        const auto rx = elem->FloatAttribute( "rx", 0.f );
        const auto ry = elem->FloatAttribute( "ry", 0.f );
        if ( rx == 0.f || ry == 0.f )
            return {};
        auto points = getEllipsePoints( {
            .cx = cx,
            .cy = cy,
            .rx = rx,
            .ry = ry,
        } );
        return { { std::move( points ) } };
    }

    Expected<Contours2f> parseLine_( XMLElement* elem ) const
    {
        const auto x1 = elem->FloatAttribute( "x1", 0.f );
        const auto y1 = elem->FloatAttribute( "y1", 0.f );
        const auto x2 = elem->FloatAttribute( "x2", 0.f );
        const auto y2 = elem->FloatAttribute( "y2", 0.f );
        Contour2f line {
            Vector2f { x1, y1 },
            Vector2f { x2, y2 },
        };
        return { { std::move( line ) } };
    }

    Expected<Contours2f> parsePolygon_( XMLElement* elem ) const
    {
        const std::string pointsStr = elem->Attribute( "points" );
        auto points = parser::parsePoints( pointsStr );
        if ( !points )
            return unexpected( std::move( points.error() ) );
        close( *points );
        return { { std::move( *points ) } };
    }

    Expected<Contours2f> parsePolyline_( XMLElement* elem ) const
    {
        const std::string pointsStr = elem->Attribute( "points" );
        auto points = parser::parsePoints( pointsStr );
        if ( !points )
            return unexpected( std::move( points.error() ) );
        return { { std::move( *points ) } };
    }

    Expected<Contours2f> parseRect_( XMLElement* elem ) const
    {
        const auto x = elem->FloatAttribute( "x", 0.f );
        const auto y = elem->FloatAttribute( "y", 0.f );
        const auto width = elem->FloatAttribute( "width", 0.f );
        const auto height = elem->FloatAttribute( "height", 0.f );
        auto rx = elem->FloatAttribute( "rx", 0.f );
        auto ry = elem->FloatAttribute( "ry", 0.f );
        if ( width == 0.f || height == 0.f )
            return {};
        if ( rx == 0.f && ry == 0.f )
        {
            Contour2f rect {
                { x, y },
                { x, y + height },
                { x + width, y + height },
                { x + width, y },
                { x, y },
            };
            return { { std::move( rect ) } };
        }
        else
        {
            // SVG requirements
            if ( rx == 0.f )
                rx = ry;
            else if ( ry == 0.f )
                ry = rx;
            if ( width / 2.f < rx )
                rx = width / 2.f;
            if ( height / 2.f < ry )
                ry = height / 2.f;

            Contour2f points;
            for ( auto p : getEllipsePoints( {
                .cx = x + width - rx,
                .cy = y + ry,
                .rx = rx,
                .ry = ry,
                .a0 = -PI2_F,
                .a1 = 0.f,
            } ) )
                points.emplace_back( p );
            for ( auto p : getEllipsePoints( {
                .cx = x + width - rx,
                .cy = y + height - ry,
                .rx = rx,
                .ry = ry,
                .a0 = 0.f,
                .a1 = PI2_F,
            } ) )
                points.emplace_back( p );
            for ( auto p : getEllipsePoints( {
                .cx = x + rx,
                .cy = y + height - ry,
                .rx = rx,
                .ry = ry,
                .a0 = PI2_F,
                .a1 = PI_F,
            } ) )
                points.emplace_back( p );
            for ( auto p : getEllipsePoints( {
                .cx = x + rx,
                .cy = y + ry,
                .rx = rx,
                .ry = ry,
                .a0 = -PI_F,
                .a1 = -PI2_F,
            } ) )
                points.emplace_back( p );
            close( points );
            return { { std::move( points ) } };
        }
    }

private:
    XMLDocument doc_;
};

} // namespace

Expected<Polyline3> fromSvg( const std::filesystem::path& file, const LinesLoadSettings& )
{
    SvgLoader loader( file );
    if ( !loader )
        return unexpected( loader.error() );

    return loader.parse().transform( [] ( Polyline2&& polyline2 )
    {
        return polyline2.toPolyline<Vector3f>();
    } );
}

Expected<Polyline3> fromSvg( std::istream& in, const LinesLoadSettings& settings )
{
    return readCharBuffer( in ).and_then( [&] ( Buffer<char>&& buf )
    {
        return fromSvg( buf.data(), buf.size(), settings );
    } );
}

Expected<Polyline3> fromSvg( const char* data, size_t size, const LinesLoadSettings& )
{
    SvgLoader loader( data, size );
    if ( !loader )
        return unexpected( loader.error() );

    return loader.parse().transform( [] ( Polyline2&& polyline2 )
    {
        return polyline2.toPolyline<Vector3f>();
    } );
}

MR_ADD_LINES_LOADER(IOFilter( "SVG (.svg)", "*.svg" ), fromSvg)

} // namespace MR::LinesLoad
#endif
