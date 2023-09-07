#include "MRMesh/MRPython.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRAffineXf2.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRId.h"
#include "MRMesh/MRLine3.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRViewportId.h"
#include "MRMesh/MREdgePoint.h"
#include "MRMesh/MRPolylineTopology.h"
#include "MRMesh/MRTriPoint.h"
#include "MRMesh/MRMeshTriPoint.h"
#include "MRMesh/MREdgePaths.h"
#include "MRMesh/MRFillContour.h"
#include "MRMesh/MRExpandShrink.h"
#include "MRMesh/MRColor.h"
#include "MRMesh/MRLineSegm.h"
#include "MRMesh/MRIntersection.h"
#include "MRMesh/MR2DContoursTriangulation.h"
#include "MRMesh/MRPointOnFace.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRExpected.h"
#include <pybind11/functional.h>
#include <pybind11/stl.h>

MR_INIT_PYTHON_MODULE( mrmeshpy )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorFloat, float )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, ExpectedVoid, []( pybind11::module_& m )\
{
    using expectedType = MR::VoidOrErrStr;
    pybind11::class_<expectedType>( m, "ExpectedVoid" ).
        def( "has_value", &expectedType::has_value ).
        def( "error", ( const std::string& ( expectedType::* )( )const& )& expectedType::error );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Path, [] ( pybind11::module_& m )
{
    pybind11::class_<std::filesystem::path>( m, "Path" ).
        def( pybind11::init( [] ( const std::string& s )
    {
        return MR::pathFromUtf8( s );
    } ) );
    pybind11::implicitly_convertible<std::string, std::filesystem::path>();
    // it could be simlier, but bug in clang 14 does not work with it:
    // def( pybind11::init<const std::u8string&>() );
    // pybind11::implicitly_convertible<std::u8string, std::filesystem::path>();
} )

#define MR_ADD_PYTHON_BOX(name, VectorType ) \
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, name, [] ( pybind11::module_& m )\
{\
    using BoxType = MR::Box<VectorType>;\
    pybind11::class_<BoxType>( m, #name, "Box given by its min- and max- corners" ).\
        def( pybind11::init<>() ).\
        def_readwrite( "min", &BoxType::min, "create invalid box by default" ).\
        def_readwrite( "max", &BoxType::max ).\
        def( "valid", &BoxType::valid ).\
        def( "center", &BoxType::center ).\
        def( "size", &BoxType::size ).\
        def( "diagonal", &BoxType::diagonal ).\
        def( "volume", &BoxType::volume ).\
        def( "include", ( void( BoxType::* )( const VectorType& ) ) &BoxType::include, pybind11::arg("pt"), "minimally increases the box to include given point" ).\
        def( "include", ( void( BoxType::* )( const BoxType& ) ) &BoxType::include, pybind11::arg("b"), "minimally increases the box to include another box" ).\
        def( "contains", &BoxType::contains, pybind11::arg( "pt" ), "checks whether given point is inside (including the surface) of the box" ).\
        def( "getBoxClosestPointTo", &BoxType::getBoxClosestPointTo, pybind11::arg( "pt" ), "returns closest point in the box to given point" ).\
        def( "intersects", &BoxType::intersects, pybind11::arg( "b" ), "checks whether this box intersects or touches given box" ).\
        def( "intersection", &BoxType::intersection, pybind11::arg( "b" ), "computes intersection between this and other box" ).\
        def( "intersect", &BoxType::intersect, pybind11::arg( "b" ), "computes intersection between this and other box" ).\
        def( "getDistanceSq", &BoxType::getDistanceSq, pybind11::arg( "b" ), "returns squared distance between this box and given one; " \
                                                                                                                    "returns zero if the boxes touch or intersect").\
        def( "insignificantlyExpanded", &BoxType::insignificantlyExpanded, "expands min and max to their closest representable value" ).\
        def( pybind11::self == pybind11::self ).\
        def( pybind11::self != pybind11::self );\
    m.def( "transformed", ( BoxType( * )( const BoxType&, const MR::AffineXf<VectorType>& ) ) &MR::transformed<VectorType>, pybind11::arg( "box" ), pybind11::arg( "xf" ),\
        "find the tightest box enclosing this one after transformation" );\
    m.def( "transformed", ( BoxType( * )( const BoxType&, const MR::AffineXf<VectorType>* ) ) &MR::transformed<VectorType>, pybind11::arg( "box" ), pybind11::arg( "xf" ),\
        "this version returns input box as is if pointer to transformation is null" );\
} )

MR_ADD_PYTHON_BOX( Box2f, MR::Vector2f )
MR_ADD_PYTHON_BOX( Box3f, MR::Vector3f )
MR_ADD_PYTHON_BOX( Box2d, MR::Vector2d )
MR_ADD_PYTHON_BOX( Box3d, MR::Vector3d )

#define MR_ADD_PYTHON_VECTOR2(name, type) \
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, name, [] ( pybind11::module_& m )\
{\
    using VectorType = MR::Vector2<type>;\
    pybind11::class_<VectorType>( m, #name, "two-dimensional vector" ).\
        def( pybind11::init<>() ).\
        def( pybind11::init<type, type>(), pybind11::arg( "x" ), pybind11::arg( "y" ) ).\
        /*def( pybind11::init<const MR::Vector2i&>(), pybind11::arg( "v" ) ).\
        def( pybind11::init<const MR::Vector2f&>(), pybind11::arg( "v" ) ).\
        def( pybind11::init<const MR::Vector2d&>(), pybind11::arg( "v" ) ).\
        def( pybind11::init<const MR::Vector3i&>(), pybind11::arg( "v" ) ).\
        def( pybind11::init<const MR::Vector3f&>(), pybind11::arg( "v" ) ).\
        def( pybind11::init<const MR::Vector3d&>(), pybind11::arg( "v" ) ).*/\
        def_readwrite( "x", &VectorType::x ).\
        def_readwrite( "y", &VectorType::y ).\
        def_static( "diagonal", &VectorType::diagonal, pybind11::arg( "a" ) ).\
        def_static( "plusX", &VectorType::plusX ).\
        def_static( "plusY", &VectorType::plusY ).\
        def_static( "minusX", &VectorType::minusX ).\
        def_static( "minusY", &VectorType::minusY ).\
        def( pybind11::self + pybind11::self ).\
        def( pybind11::self - pybind11::self ).\
        def( pybind11::self * type() ).\
        def( type() * pybind11::self ).\
        def( pybind11::self / type() ).\
        def( pybind11::self += pybind11::self ).\
        def( pybind11::self -= pybind11::self ).\
        def( pybind11::self *= type() ).\
        def( pybind11::self /= type() ).\
        def( -pybind11::self ).\
        def( pybind11::self == pybind11::self ).\
        def( "length", &VectorType::length ).\
        def( "lengthSq", &VectorType::lengthSq ).\
        def( "normalized", &VectorType::normalized ).\
        def( "__repr__", [](const VectorType& data){\
            std::stringstream ss;\
            ss << #name << "[" << data.x << ", " << data.y << "]";\
            return ss.str();\
        } );\
    m.def( "dot", ( type( * )( const VectorType&, const VectorType& ) )& MR::dot<type>, pybind11::arg( "a" ), pybind11::arg( "b" ), "dot product" );\
    m.def( "cross", ( type( * )( const VectorType&, const VectorType& ) )& MR::cross<type>, pybind11::arg( "a" ), pybind11::arg( "b" ), "cross product" );\
} )

MR_ADD_PYTHON_VECTOR2( Vector2i, int )
MR_ADD_PYTHON_VECTOR2( Vector2f, float )
MR_ADD_PYTHON_VECTOR2( Vector2d, double )

MR_ADD_PYTHON_VEC( mrmeshpy, Contour2f, MR::Vector2f )
MR_ADD_PYTHON_VEC( mrmeshpy, Contours2f, MR::Contour2f )
MR_ADD_PYTHON_VEC( mrmeshpy, Contour2d, MR::Vector2d )
MR_ADD_PYTHON_VEC( mrmeshpy, Contours2d, MR::Contour2d )

#define MR_ADD_PYTHON_VECTOR3(name, type) \
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, name, [] ( pybind11::module_& m )\
{\
    using VectorType = MR::Vector3<type>;\
    auto vectorClass = pybind11::class_<VectorType>( m, #name, "three-dimensional vector" ).\
        def( pybind11::init<>() ).\
        def( pybind11::init<type, type, type>(), pybind11::arg( "x" ), pybind11::arg( "y" ), pybind11::arg( "z" ) ).\
        /*def( pybind11::init<const MR::Vector2i&>(), pybind11::arg( "v" ) ).\
        def( pybind11::init<const MR::Vector2f&>(), pybind11::arg( "v" ) ).\
        def( pybind11::init<const MR::Vector2d&>(), pybind11::arg( "v" ) ).\
        def( pybind11::init<const MR::Vector3i&>(), pybind11::arg( "v" ) ).\
        def( pybind11::init<const MR::Vector3f&>(), pybind11::arg( "v" ) ).\
        def( pybind11::init<const MR::Vector3d&>(), pybind11::arg( "v" ) ).*/\
        def_readwrite( "x", &VectorType::x ).\
        def_readwrite( "y", &VectorType::y ).\
        def_readwrite( "z", &VectorType::z ).\
        def( pybind11::self + pybind11::self ).\
        def( pybind11::self - pybind11::self ).\
        def( pybind11::self * type() ).\
        def( type() * pybind11::self ).\
        def( pybind11::self / type() ).\
        def( pybind11::self += pybind11::self ).\
        def( pybind11::self -= pybind11::self ).\
        def( pybind11::self *= type() ).\
        def( pybind11::self /= type() ).\
        def( -pybind11::self ).\
        def( pybind11::self == pybind11::self ).\
        def_static( "diagonal", &VectorType::diagonal ).\
        def( "lengthSq", &VectorType::lengthSq ).\
        def( "__repr__", [](const VectorType& data){\
            std::stringstream ss;\
            ss << #name << "[" << data.x << ", " << data.y << ", " << data.z << "]";\
            return ss.str();\
        } );\
    if constexpr ( !std::is_same_v<type, int> ) \
    {\
        vectorClass.def( "length", &VectorType::length ).\
        def( "normalized", &VectorType::normalized );\
        m.def( "angle", ( type( * )( const VectorType&, const VectorType& ) )& MR::angle<type>,\
            pybind11::arg( "a" ), pybind11::arg( "b" ), "angle in radians between two vectors" );\
    }\
\
    m.def( "dot", ( type( * )( const VectorType&, const VectorType& ) )& MR::dot<type>, pybind11::arg( "a" ), pybind11::arg( "b" ), "dot product" );\
    m.def( "cross", ( VectorType( * )( const VectorType&, const VectorType& ) )& MR::cross<type>, pybind11::arg( "a" ), pybind11::arg( "b" ), "cross product" );\
    m.def( "mixed", ( type( * )( const VectorType&, const VectorType&, const VectorType& ) )& MR::mixed<type>,\
        pybind11::arg( "a" ), pybind11::arg( "b" ),pybind11::arg( "c" ), "mixed product" );\
    m.def( "mult", ( VectorType( * )( const VectorType&, const VectorType& ) )& MR::mult<type>, pybind11::arg( "a" ), pybind11::arg( "b" ), "per component multiplication" );\
} )

MR_ADD_PYTHON_VECTOR3( Vector3i, int )
MR_ADD_PYTHON_VECTOR3( Vector3f, float )
MR_ADD_PYTHON_VECTOR3( Vector3d, double )

MR_ADD_PYTHON_VEC( mrmeshpy, Contour3f, MR::Vector3f )
MR_ADD_PYTHON_VEC( mrmeshpy, Contours3f, MR::Contour3f )
MR_ADD_PYTHON_VEC( mrmeshpy, Contour3d, MR::Vector3d )
MR_ADD_PYTHON_VEC( mrmeshpy, Contours3d, MR::Contour3d )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Color, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::Color>( m, "Color" ).
        def( pybind11::init<>() ).
        def( pybind11::init<int, int, int, int>(),
            pybind11::arg( "r" ), pybind11::arg( "g" ), pybind11::arg( "b" ), pybind11::arg( "a" ) = 255 ).
        def( pybind11::init<float, float, float, float>(),
            pybind11::arg( "r" ), pybind11::arg( "g" ), pybind11::arg( "b" ), pybind11::arg( "a" ) = 1.0f ).
        def_readwrite( "r", &MR::Color::r ).
        def_readwrite( "g", &MR::Color::g ).
        def_readwrite( "b", &MR::Color::b ).
        def_readwrite( "a", &MR::Color::a ).
        def( "__repr__", [] ( const MR::Color& data )
        {
            std::stringstream ss;
            ss << "Color[" << data.r << ", " << data.g << ", " << data.b << ", " << data.a << "]";
            return ss.str();
        } );
} )
MR_ADD_PYTHON_VEC( mrmeshpy, vectorColor, MR::Color )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Matrix3f, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::Matrix3f>( m, "Matrix3f", "arbitrary 3x3 matrix" ).
        def( pybind11::init<>() ).
        def_readwrite( "x", &MR::Matrix3f::x, "rows, identity matrix by default" ).
        def_readwrite( "y", &MR::Matrix3f::y ).
        def_readwrite( "z", &MR::Matrix3f::z ).
        def_static( "zero", &MR::Matrix3f::zero ).
        def_static( "scale", ( MR::Matrix3f( * )( float ) noexcept )& MR::Matrix3f::scale, pybind11::arg( "s" ), "returns a matrix that scales uniformly" ).
        def_static( "scale", ( MR::Matrix3f( * )( float, float, float ) noexcept )& MR::Matrix3f::scale,
            pybind11::arg( "x" ), pybind11::arg( "y" ), pybind11::arg( "z" ), "returns a matrix that has its own scale along each axis" ).
        def_static( "rotation", ( MR::Matrix3f( * )( const MR::Vector3f&, float ) noexcept )& MR::Matrix3f::rotation,
            pybind11::arg( "axis" ), pybind11::arg( "angle" ),"creates matrix representing rotation around given axis on given angle").
        def_static( "rotation", ( MR::Matrix3f( * )( const MR::Vector3f&, const MR::Vector3f& ) noexcept )& MR::Matrix3f::rotation,
            pybind11::arg( "from" ), pybind11::arg( "to" ), "creates matrix representing rotation that after application to (from) makes (to) vector" ).
        def_static( "rotationFromEuler", &MR::Matrix3f::rotationFromEuler, pybind11::arg( "eulerAngles" ), 
            "creates matrix representing rotation from 3 Euler angles: R=R(z)*R(y)*R(x)\n"
            "see more https://en.wikipedia.org/wiki/Euler_angles#Conventions_by_intrinsic_rotations" ).
        def( "normSq", &MR::Matrix3f::normSq, "compute sum of squared matrix elements" ).
        def( "norm", &MR::Matrix3f::norm ).
        def( "det", &MR::Matrix3f::det, "computes determinant of the matrix" ).
        def( "inverse", &MR::Matrix3f::inverse, "computes inverse matrix" ).
        def( "transposed", &MR::Matrix3f::transposed, "computes transposed matrix" ).
        def( "toEulerAngles", &MR::Matrix3f::toEulerAngles, "returns 3 Euler angles, assuming this is a rotation matrix composed as follows: R=R(z)*R(y)*R(x)" ).
        def( pybind11::self + pybind11::self ).
        def( pybind11::self - pybind11::self ).
        def( pybind11::self* float() ).
        def( pybind11::self* MR::Vector3f() ).
        def( pybind11::self* pybind11::self ).
        def( float()* pybind11::self ).
        def( pybind11::self / float() ).
        def( pybind11::self += pybind11::self ).
        def( pybind11::self -= pybind11::self ).
        def( pybind11::self *= float() ).
        def( pybind11::self /= float() ).
        def( pybind11::self == pybind11::self );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, LineSegm, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::LineSegm2f>( m, "LineSegm2f", "a segment of 2-dimensional line" ).
        def( pybind11::init<>() ).
        def( pybind11::init<const MR::Vector2f&, const MR::Vector2f&>() ).
        def_readwrite( "a", &MR::LineSegm2f::a ).
        def_readwrite( "b", &MR::LineSegm2f::b );

    pybind11::class_<MR::LineSegm3f>( m, "LineSegm3f", "a segment of 3-dimensional line" ).
        def( pybind11::init<>() ).
        def( pybind11::init<const MR::Vector3f&, const MR::Vector3f&>() ).
        def_readwrite( "a", &MR::LineSegm3f::a ).
        def_readwrite( "b", &MR::LineSegm3f::b );

    m.def( "intersection", ( std::optional<MR::Vector2f>( * )( const MR::LineSegm2f&, const MR::LineSegm2f& ) )& MR::intersection,
        pybind11::arg( "segm1" ), pybind11::arg( "segm2" ),
        "finds an intersection between a segm1 and a segm2\n"
        "return null if they don't intersect (even if they match)" );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, PointOnFace, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::PointOnFace>( m, "PointOnFace", "point located on some mesh face" ).
        def( pybind11::init<>() ).
        def_readwrite( "face", &MR::PointOnFace::face ).
        def_readwrite( "point", &MR::PointOnFace::point );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, AffineXf3f, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::AffineXf3f>( m, "AffineXf3f", "affine transformation: y = A*x + b, where A in VxV, and b in V" ).
        def( pybind11::init<>() ).
        def_readwrite( "A", &MR::AffineXf3f::A ).
        def_readwrite( "b", &MR::AffineXf3f::b ).
        def_static( "translation", &MR::AffineXf3f::translation, pybind11::arg( "b" ), "creates translation-only transformation (with identity linear component)" ).
        def_static( "linear", &MR::AffineXf3f::linear, pybind11::arg( "A" ), "creates linear-only transformation (without translation)" ).
        def_static( "xfAround", &MR::AffineXf3f::xfAround, pybind11::arg( "A" ), pybind11::arg( "stable" ), "creates transformation with given linear part with given stable point" ).
        def( "linearOnly", &MR::AffineXf3f::linearOnly, pybind11::arg( "x" ),
            "applies only linear part of the transformation to given vector (e.g. to normal) skipping adding shift (b)\n"
            "for example if this is a rigid transformation, then only rotates input vector" ).
        def( "inverse", &MR::AffineXf3f::inverse, "computes inverse transformation" ).
        def( "__call__", &MR::AffineXf3f::operator(), "application of the transformation to a point" ).
        def( pybind11::self* pybind11::self ).
        def( pybind11::self == pybind11::self );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Line3, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::Line3f>( m, "Line3f", "3-dimensional line: cross( x - p, d ) = 0" ).
        def( pybind11::init<>() ).
        def( pybind11::init<const MR::Vector3f&, const MR::Vector3f&>(), pybind11::arg( "p" ), pybind11::arg( "d" ) ).
        def_readwrite( "p", &MR::Line3f::p ).
        def_readwrite( "d", &MR::Line3f::d ).
        def( "distanceSq", &MR::Line3f::distanceSq, pybind11::arg( "x" ), "returns squared distance from given point to this line" ).
        def( "normalized", &MR::Line3f::normalized, "returns same line represented with unit d-vector" ).
        def( "project", &MR::Line3f::project, pybind11::arg( "x" ), "finds the closest point on line" );

    pybind11::class_<MR::Line3d>( m, "Line3d", "3-dimensional line: cross( x - p, d ) = 0" ).
        def( pybind11::init<>() ).
        def( pybind11::init<const MR::Vector3d&, const MR::Vector3d&>(), pybind11::arg( "p" ), pybind11::arg( "d" ) ).
        def_readwrite( "p", &MR::Line3d::p ).
        def_readwrite( "d", &MR::Line3d::d ).
        def( "distanceSq", &MR::Line3d::distanceSq, pybind11::arg( "x" ), "returns squared distance from given point to this line" ).
        def( "normalized", &MR::Line3d::normalized, "returns same line represented with unit d-vector" ).
        def( "project", &MR::Line3d::project, pybind11::arg( "x" ), "finds the closest point on line" );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Plane3f, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::Plane3f>( m, "Plane3f", "3-dimensional plane: dot(n,x) - d = 0" ).
        def( pybind11::init<>() ).
        def_readwrite( "n", &MR::Plane3f::n ).
        def_readwrite( "d", &MR::Plane3f::d );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, FaceId, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::FaceId>( m, "FaceId" ).
        def( pybind11::init<>() ).
        def( pybind11::init<int>() ).
        def( "valid", &MR::FaceId::valid ).
        def( "get", &MR::FaceId::operator int );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, VertId, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::VertId>( m, "VertId" ).
        def( pybind11::init<>() ).
        def( pybind11::init<int>() ).
        def( "valid", &MR::VertId::valid ).
        def( "get", &MR::VertId::operator int );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, UndirectedEdgeId, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::UndirectedEdgeId>( m, "UndirectedEdgeId" ).
        def( pybind11::init<>() ).
        def( pybind11::init<int>() ).
        def( "valid", &MR::UndirectedEdgeId::valid ).
        def( "get", &MR::UndirectedEdgeId::operator int );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, ViewportId, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::ViewportId>( m, "ViewportId",
        "stores unique identifier of a viewport, which is power of two;\n"
        "id=0 has a special meaning of default viewport in some contexts" ).
        def( pybind11::init<>() ).
        def( pybind11::init<unsigned>() ).
        def( "value", &MR::ViewportId::value ).
        def( "valid", &MR::ViewportId::valid );

    pybind11::class_<MR::ViewportMask>( m, "ViewportMask", "stores mask of viewport unique identifiers" ).
        def( pybind11::init<>() ).
        def( pybind11::init<unsigned>() ).
        def( pybind11::init<MR::ViewportId>() ).
        def_static( "all", &MR::ViewportMask::all, "mask meaning all or any viewports" ).
        def_static( "any", &MR::ViewportMask::any, "mask meaning all or any viewports" );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshPoint, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::EdgePoint>( m, "EdgePoint", "encodes a point on a mesh edge" ).
        def( pybind11::init<>() ).
        def( pybind11::init<MR::EdgeId, float>(), pybind11::arg( "e" ), pybind11::arg( "a" ) ).
        def_readwrite( "e", &MR::EdgePoint::e ).
        def_readwrite( "a", &MR::EdgePoint::a, "a in [0,1], a=0 => point is in org( e ), a=1 => point is in dest( e )" ).
        def( "inVertex", ( MR::VertId( MR::EdgePoint::* )( const MR::MeshTopology& )const )& MR::EdgePoint::inVertex,
            pybind11::arg( "topology" ), "returns valid vertex id if the point is in vertex, otherwise returns invalid id" ).
        def( "inVertex", ( MR::VertId( MR::EdgePoint::* )( const MR::PolylineTopology& )const )& MR::EdgePoint::inVertex,
            pybind11::arg( "topology" ), "returns valid vertex id if the point is in vertex, otherwise returns invalid id" ).
        def( "inVertex", ( bool( MR::EdgePoint::* )( )const )& MR::EdgePoint::inVertex, "returns true if the point is in a vertex" ).
        def( "getClosestVertex", ( MR::VertId( MR::EdgePoint::* )( const MR::MeshTopology & ) const )& MR::EdgePoint::getClosestVertex, pybind11::arg( "topology" ), "returns one of two edge vertices, closest to this point" ).
        def( "getClosestVertex", ( MR::VertId( MR::EdgePoint::* )( const MR::PolylineTopology & ) const )& MR::EdgePoint::getClosestVertex, pybind11::arg( "topology" ), "returns one of two edge vertices, closest to this point" ).
        def( "sym", &MR::EdgePoint::sym, "represents the same point relative to sym edge in" ).
        def( pybind11::self == pybind11::self );

    pybind11::class_<MR::TriPointf>( m, "TriPointf", 
        "encodes a point inside a triangle using barycentric coordinates\n"
        "\tNotations used below: v0, v1, v2 - points of the triangle").
        def( pybind11::init<>() ).
        def( pybind11::init<float, float>(), pybind11::arg( "a" ), pybind11::arg( "b" ) ).
        def( pybind11::init<const MR::Vector3f&, const MR::Vector3f&, const MR::Vector3f&, const MR::Vector3f&>(),
            pybind11::arg( "p" ), pybind11::arg( "v0" ), pybind11::arg( "v1" ), pybind11::arg( "v2" ), 
            "given a point coordinates and triangle (v0,v1,v2) computes barycentric coordinates of the point" ).
        def( pybind11::init<const MR::Vector3f&, const MR::Vector3f&, const MR::Vector3f&>(),
            pybind11::arg( "p" ), pybind11::arg( "v1" ), pybind11::arg( "v2" ),
            "given a point coordinates and triangle (0,v1,v2) computes barycentric coordinates of the point" ).
        def_readwrite( "a", &MR::TriPointf::a, 
            "barycentric coordinates:\n"
            "\ta+b in [0,1], a+b=0 => point is in v0, a+b=1 => point is on [v1,v2] edge\n"
            "a in [0,1], a=0 => point is on [v2,v0] edge, a=1 => point is in v1" ).
        def_readwrite( "b", &MR::TriPointf::b, "b in [0,1], b=0 => point is on [v0,v1] edge, b=1 => point is in v2" );

    pybind11::class_<MR::MeshTriPoint>( m, "MeshTriPoint",
        "encodes a point inside a triangular mesh face using barycentric coordinates\n"
        "\tNotations used below:\n" 
        "\t v0 - the value in org( e )\n"
        "\t v1 - the value in dest( e )\n"
        "\t v2 - the value in dest( next( e ) )" ).
        def( pybind11::init<>() ).
        def( pybind11::init<MR::EdgeId, MR::TriPointf>(), pybind11::arg( "e" ), pybind11::arg( "bary" ) ).
        def( pybind11::init<const MR::EdgePoint&>(), pybind11::arg( "ep" ) ).
        def( pybind11::init<MR::EdgeId, const MR::Vector3f&, const MR::Vector3f&, const MR::Vector3f&, const MR::Vector3f&>(),
            pybind11::arg( "e" ), pybind11::arg( "p" ), pybind11::arg( "v0" ), pybind11::arg( "v1" ), pybind11::arg( "v2" ),
            "given a point coordinates computes its barycentric coordinates" ).
        def_readwrite( "e", &MR::MeshTriPoint::e, "left face of this edge is considered" ).
        def_readwrite( "bary", &MR::MeshTriPoint::bary,
            "barycentric coordinates\n"
            "a in [0,1], a=0 => point is on next( e ) edge, a=1 => point is in dest( e )\n"
            "b in [0,1], b=0 => point is on e edge, b=1 => point is in dest( next( e ) )\n"
            "a+b in [0,1], a+b=0 => point is in org( e ), a+b=1 => point is on prev( e.sym() ) edge" ).
        def( "onEdge", &MR::MeshTriPoint::onEdge, pybind11::arg( "topology" ), "returns valid value if the point is on edge, otherwise returns null optional" ).
        def( "inVertex", ( MR::VertId( MR::MeshTriPoint::* )( const MR::MeshTopology& )const )& MR::MeshTriPoint::inVertex,
            pybind11::arg( "topology" ), "returns valid vertex id if the point is in vertex, otherwise returns invalid id" ).
        def( "inVertex", ( bool( MR::MeshTriPoint::* )( )const )& MR::MeshTriPoint::inVertex, "returns true if the point is in a vertex" ).
        def( "isBd", &MR::MeshTriPoint::isBd, pybind11::arg( "topology" ), pybind11::arg( "region" ) = nullptr, "returns true if the point is in vertex on on edge, and that location is on the boundary of the region" );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, EdgeId, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::EdgeId>( m, "EdgeId" ).
        def( pybind11::init<>() ).
        def( pybind11::init<int>() ).
        def( pybind11::init<MR::UndirectedEdgeId>() ).
        def( "valid", &MR::EdgeId::valid ).
        def( "sym", &MR::EdgeId::sym, "returns identifier of the edge with same ends but opposite orientation" ).
        def( "undirected", &MR::EdgeId::undirected, "returns unique identifier of the edge ignoring its direction" ).
        def( "get", &MR::EdgeId::operator int );
} )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorUndirectedEdges, MR::UndirectedEdgeId )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorEdges, MR::EdgeId )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorVerts, MR::VertId )

MR_ADD_PYTHON_VEC( mrmeshpy, HolesVertIds, MR::PlanarTriangulation::HoleVertIds )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorFaces, MR::FaceId )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorEdgePath, MR::EdgePath )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, BoostBitSet, [] ( pybind11::module_& m )
{
    using type = boost::dynamic_bitset<uint64_t>;
    pybind11::class_<type>( m, "BoostBitSet" ).
        def( "size", &type::size ).
        def( "count", &type::count );
} )

#define  ADD_PYTHON_BITSET(name,type)\
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, name, [] (pybind11::module_& m)\
{\
    pybind11::class_<type,boost::dynamic_bitset<uint64_t>>( m, #name ).\
        def( pybind11::init<>() ).\
        def( "test", &type::test ).\
        def( "resize", &type::resize ).\
        def( "set",( type& ( type::* )( type::IndexType, bool ) )& type::set, pybind11::return_value_policy::reference ).\
        def( "flip",( type& ( type::* )() )& type::flip, pybind11::return_value_policy::reference ).\
        def( pybind11::self & pybind11::self ).\
        def( pybind11::self | pybind11::self ).\
        def( pybind11::self ^ pybind11::self ).\
        def( pybind11::self - pybind11::self ).\
        def( pybind11::self &= pybind11::self ).\
        def( pybind11::self |= pybind11::self ).\
        def( pybind11::self ^= pybind11::self ).\
        def( pybind11::self -= pybind11::self );\
} )

ADD_PYTHON_BITSET( VertBitSet, MR::VertBitSet )

ADD_PYTHON_BITSET( EdgeBitSet, MR::EdgeBitSet )

ADD_PYTHON_BITSET( FaceBitSet, MR::FaceBitSet )
