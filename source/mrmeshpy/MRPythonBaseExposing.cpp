#include "MRPython/MRPython.h"
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
#pragma warning(push)
#pragma warning(disable: 4464) // relative include path contains '..'
#include <pybind11/stl/filesystem.h>
#pragma warning(pop)

MR_INIT_PYTHON_MODULE( mrmeshpy )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorFloat, float )

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, ExpectedVoid, MR::Expected<void> )
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, ExpectedVoid, [] ( pybind11::module_& )
{
    using expectedType = MR::Expected<void>;
    MR_PYTHON_CUSTOM_CLASS( ExpectedVoid ).
        def( "has_value", &expectedType::has_value ).
        def( "error", ( const std::string& ( expectedType::* )( )const& )& expectedType::error );
} )

class DeprecatedPath
{
public:
    DeprecatedPath( const std::filesystem::path& path ) : path_{ path }
    {
        PyErr_WarnEx( PyExc_DeprecationWarning, "mrmeshpy.Path is deprecated, use os.PathLike type instead", 1 );
    }
    operator std::string () const { return MR::utf8string( path_ ); }
private:
    std::filesystem::path path_;
};

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, Path, DeprecatedPath )
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Path, [] ( pybind11::module_& )
{
    MR_PYTHON_CUSTOM_CLASS( Path ).
        def( pybind11::init<const std::filesystem::path&>(), pybind11::arg( "path" ) ).
        def( "__fspath__", &DeprecatedPath::operator std::string );
} )

#define MR_ADD_PYTHON_BOX( name, VectorType ) \
MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, name, MR::Box<VectorType> ) \
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, name, [] ( pybind11::module_& m )      \
{\
    using BoxType = MR::Box<VectorType>;      \
    using ValueType = typename VectorType::ValueType;  \
    MR_PYTHON_CUSTOM_CLASS( name ).doc() =                               \
        "Box given by its min- and max- corners";                      \
    MR_PYTHON_CUSTOM_CLASS( name ).                                      \
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
        def( "getDistanceSq", ( ValueType( BoxType::* )( const BoxType& ) const ) &BoxType::getDistanceSq, pybind11::arg( "b" ), "returns squared distance between this box and given one; " \
                                                                                                                    "returns zero if the boxes touch or intersect").\
        def( "getDistanceSq", ( ValueType( BoxType::* )( const VectorType& ) const ) &BoxType::getDistanceSq, pybind11::arg( "pt" ), \
            "returns squared distance between this box and given point; returns zero if the point is inside or on the boundary of the box").\
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
MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, name, MR::Vector2<type> ) \
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, name, [] ( pybind11::module_& m ) \
{\
    using VectorType = MR::Vector2<type>;\
    MR_PYTHON_CUSTOM_CLASS( name ).doc() = "two-dimensional vector";\
    MR_PYTHON_CUSTOM_CLASS( name ).\
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
        def( "__repr__", [](const VectorType& data){\
            std::stringstream ss;\
            ss << #name << "[" << data.x << ", " << data.y << "]";\
            return ss.str();\
        } ).\
         def( "__iter__", [](VectorType& data) {\
            return pybind11::make_iterator<pybind11::return_value_policy::reference_internal>( begin( data ), end( data ) );\
        }, pybind11::keep_alive<0, 1>() );\
    m.def( "dot", ( type( * )( const VectorType&, const VectorType& ) )& MR::dot<type>, pybind11::arg( "a" ), pybind11::arg( "b" ), "dot product" );\
    m.def( "cross", ( type( * )( const VectorType&, const VectorType& ) )& MR::cross<type>, pybind11::arg( "a" ), pybind11::arg( "b" ), "cross product" );\
    \
    /* Need to wrap in a template lambda for `if constexpr` to work properly. */\
    []<typename T, typename V>(){ \
        if constexpr ( std::is_floating_point_v<T> ) \
        { \
            MR_PYTHON_CUSTOM_CLASS( name ) \
                .def( "normalized", &V::normalized ) \
            ; \
        } \
    }.template operator()<type, VectorType>(); \
} )

MR_ADD_PYTHON_VECTOR2( Vector2i, int )
MR_ADD_PYTHON_VECTOR2( Vector2f, float )
MR_ADD_PYTHON_VECTOR2( Vector2d, double )

MR_ADD_PYTHON_VEC( mrmeshpy, Contour2f, MR::Vector2f )
MR_ADD_PYTHON_VEC( mrmeshpy, Contours2f, MR::Contour2f )
MR_ADD_PYTHON_VEC( mrmeshpy, Contour2d, MR::Vector2d )
MR_ADD_PYTHON_VEC( mrmeshpy, Contours2d, MR::Contour2d )

#define MR_ADD_PYTHON_VECTOR3(name, type)\
MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, name, MR::Vector3<type> )\
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, name, [] ( pybind11::module_& m )\
{\
    using VectorType = MR::Vector3<type>;\
    MR_PYTHON_CUSTOM_CLASS( name ).doc() = "three-dimensional vector";\
    MR_PYTHON_CUSTOM_CLASS( name ).\
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
        } ).\
        def( "__iter__", [](VectorType& data) {\
            return pybind11::make_iterator<pybind11::return_value_policy::reference_internal>( begin( data ), end( data ) );\
        }, pybind11::keep_alive<0, 1>() );\
    [&]<typename T = type>() \
    {\
        if constexpr ( !std::is_same_v<T, int> ) \
        { \
            MR_PYTHON_CUSTOM_CLASS( name ).\
                def( "length", &MR::Vector3<T>::length ).\
                def( "normalized", &MR::Vector3<T>::normalized );\
            m.def( "angle", ( type( * )( const MR::Vector3<T>&, const MR::Vector3<T>& ) )& MR::angle<type>,\
                pybind11::arg( "a" ), pybind11::arg( "b" ), "angle in radians between two vectors" );\
        } \
    }();\
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

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, Color, MR::Color )
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Color, [] ( pybind11::module_& )
{
    MR_PYTHON_CUSTOM_CLASS( Color ).
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

#define MR_ADD_PYTHON_MATRIX3( name, type ) \
MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, name, type ) \
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, name, [] ( pybind11::module_& ) \
{ \
    using ValueType = typename type::ValueType; \
    using VectorType = typename type::VectorType;       \
    MR_PYTHON_CUSTOM_CLASS( name ).doc() = \
        "arbitrary 3x3 matrix"; \
    MR_PYTHON_CUSTOM_CLASS( name ). \
        def( pybind11::init<>() ). \
        def_readwrite( "x", &type::x, "rows, identity matrix by default" ). \
        def_readwrite( "y", &type::y ). \
        def_readwrite( "z", &type::z ). \
        def_static( "zero", &type::zero ). \
        def_static( "scale", ( type( * )( ValueType ) noexcept )& type::scale, pybind11::arg( "s" ), "returns a matrix that scales uniformly" ). \
        def_static( "scale", ( type( * )( ValueType, ValueType, ValueType ) noexcept )& type::scale, \
            pybind11::arg( "x" ), pybind11::arg( "y" ), pybind11::arg( "z" ), "returns a matrix that has its own scale along each axis" ). \
        def_static( "rotation", ( type( * )( const VectorType&, ValueType ) noexcept )& type::rotation, \
            pybind11::arg( "axis" ), pybind11::arg( "angle" ),"creates matrix representing rotation around given axis on given angle"). \
        def_static( "rotation", ( type( * )( const VectorType&, const VectorType& ) noexcept )& type::rotation, \
            pybind11::arg( "from" ), pybind11::arg( "to" ), "creates matrix representing rotation that after application to (from) makes (to) vector" ). \
        def_static( "rotationFromEuler", &type::rotationFromEuler, pybind11::arg( "eulerAngles" ),  \
            "creates matrix representing rotation from 3 Euler angles: R=R(z)*R(y)*R(x)\n" \
            "see more https://en.wikipedia.org/wiki/Euler_angles#Conventions_by_intrinsic_rotations" ). \
        def( "normSq", &type::normSq, "compute sum of squared matrix elements" ). \
        def( "norm", &type::norm ). \
        def( "det", &type::det, "computes determinant of the matrix" ). \
        def( "inverse", &type::inverse, "computes inverse matrix" ). \
        def( "transposed", &type::transposed, "computes transposed matrix" ). \
        def( "toEulerAngles", &type::toEulerAngles, "returns 3 Euler angles, assuming this is a rotation matrix composed as follows: R=R(z)*R(y)*R(x)" ). \
        def( pybind11::self + pybind11::self ). \
        def( pybind11::self - pybind11::self ). \
        def( pybind11::self * ValueType() ). \
        def( pybind11::self * VectorType() ). \
        def( pybind11::self * pybind11::self ). \
        def( ValueType() * pybind11::self ). \
        def( pybind11::self / ValueType() ). \
        def( pybind11::self += pybind11::self ). \
        def( pybind11::self -= pybind11::self ). \
        def( pybind11::self *= ValueType() ). \
        def( pybind11::self /= ValueType() ). \
        def( pybind11::self == pybind11::self ); \
} )

MR_ADD_PYTHON_MATRIX3( Matrix3f, MR::Matrix3f )
MR_ADD_PYTHON_MATRIX3( Matrix3d, MR::Matrix3d )

#define MR_ADD_PYTHON_MATRIX2( name, type ) \
MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, name, type ) \
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, name, [] ( pybind11::module_& ) \
{ \
    using ValueType = typename type::ValueType; \
    using VectorType = typename type::VectorType; \
    MR_PYTHON_CUSTOM_CLASS( name ).doc() = \
        "arbitrary 2x2 matrix"; \
    MR_PYTHON_CUSTOM_CLASS( name ). \
        def( pybind11::init<>() ). \
        def_readwrite( "x", &type::x, "rows, identity matrix by default" ). \
        def_readwrite( "y", &type::y ). \
        def_static( "zero", &type::zero ). \
        def_static( "scale", ( type( * )( ValueType ) noexcept )& type::scale, pybind11::arg( "s" ), "returns a matrix that scales uniformly" ). \
        def_static( "scale", ( type( * )( ValueType, ValueType ) noexcept )& type::scale, \
                    pybind11::arg( "x" ), pybind11::arg( "y" ), "returns a matrix that has its own scale along each axis" ). \
        def_static( "rotation", ( type( * )( ValueType ) noexcept )& type::rotation, \
                    pybind11::arg( "angle" ),"creates matrix representing rotation around origin on given angle"). \
        def_static( "rotation", ( type( * )( const VectorType&, const VectorType& ) noexcept )& type::rotation, \
                    pybind11::arg( "from" ), pybind11::arg( "to" ), "creates matrix representing rotation that after application to (from) makes (to) vector" ). \
        def( "normSq", &type::normSq, "compute sum of squared matrix elements" ). \
        def( "norm", &type::norm ). \
        def( "det", &type::det, "computes determinant of the matrix" ). \
        def( "inverse", &type::inverse, "computes inverse matrix" ). \
        def( "transposed", &type::transposed, "computes transposed matrix" ). \
        def( pybind11::self + pybind11::self ). \
        def( pybind11::self - pybind11::self ). \
        def( pybind11::self * ValueType() ). \
        def( pybind11::self * VectorType() ). \
        def( pybind11::self * pybind11::self ). \
        def( ValueType() * pybind11::self ). \
        def( pybind11::self / ValueType() ). \
        def( pybind11::self += pybind11::self ). \
        def( pybind11::self -= pybind11::self ). \
        def( pybind11::self *= ValueType() ). \
        def( pybind11::self /= ValueType() ). \
        def( pybind11::self == pybind11::self ); \
} )

MR_ADD_PYTHON_MATRIX2( Matrix2f, MR::Matrix2f )
MR_ADD_PYTHON_MATRIX2( Matrix2d, MR::Matrix2d )

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, LineSegm2f, MR::LineSegm2f )
MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, LineSegm3f, MR::LineSegm3f )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, LineSegm, [] ( pybind11::module_& m )
{
    MR_PYTHON_CUSTOM_CLASS( LineSegm2f ).doc() =
        "a segment of 2-dimensional line";
    MR_PYTHON_CUSTOM_CLASS( LineSegm2f ).
        def( pybind11::init<>() ).
        def( pybind11::init<const MR::Vector2f&, const MR::Vector2f&>() ).
        def_readwrite( "a", &MR::LineSegm2f::a ).
        def_readwrite( "b", &MR::LineSegm2f::b );

    MR_PYTHON_CUSTOM_CLASS( LineSegm3f ).doc() =
        "a segment of 3-dimensional line";
    MR_PYTHON_CUSTOM_CLASS( LineSegm3f ).
        def( pybind11::init<>() ).
        def( pybind11::init<const MR::Vector3f&, const MR::Vector3f&>() ).
        def_readwrite( "a", &MR::LineSegm3f::a ).
        def_readwrite( "b", &MR::LineSegm3f::b );

    m.def( "intersection", ( std::optional<MR::Vector2f>( * )( const MR::LineSegm2f&, const MR::LineSegm2f& ) )& MR::intersection,
        pybind11::arg( "segm1" ), pybind11::arg( "segm2" ),
        "finds an intersection between a segm1 and a segm2\n"
        "return null if they don't intersect (even if they match)" );
} )

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, PointOnFace, MR::PointOnFace )
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, PointOnFace, [] ( pybind11::module_& )
{
    MR_PYTHON_CUSTOM_CLASS( PointOnFace ).doc() =
        "point located on some mesh face";
    MR_PYTHON_CUSTOM_CLASS( PointOnFace ).
        def( pybind11::init<>() ).
        def_readwrite( "face", &MR::PointOnFace::face ).
        def_readwrite( "point", &MR::PointOnFace::point );
} )

#define MR_ADD_PYTHON_AFFINE_XF( name ) \
MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, name, MR::name ) \
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, name, [] ( pybind11::module_& ) \
{                                       \
    using AffineXfType = MR::name;      \
    MR_PYTHON_CUSTOM_CLASS( name ).doc() = "affine transformation: y = A*x + b, where A in VxV, and b in V";       \
    MR_PYTHON_CUSTOM_CLASS( name ).                                \
        def( pybind11::init<>() ).      \
        def_readwrite( "A", &AffineXfType::A ).                                         \
        def_readwrite( "b", &AffineXfType::b ).                                         \
        def_static( "translation", &AffineXfType::translation, pybind11::arg( "b" ), "creates translation-only transformation (with identity linear component)" ). \
        def_static( "linear", &AffineXfType::linear, pybind11::arg( "A" ), "creates linear-only transformation (without translation)" ).                           \
        def_static( "xfAround", &AffineXfType::xfAround, pybind11::arg( "A" ), pybind11::arg( "stable" ), "creates transformation with given linear part with given stable point" ). \
        def( "linearOnly", &AffineXfType::linearOnly, pybind11::arg( "x" ),                                                                                        \
            "applies only linear part of the transformation to given vector (e.g. to normal) skipping adding shift (b)\n"                                          \
            "for example if this is a rigid transformation, then only rotates input vector" ).                                                                     \
        def( "inverse", &AffineXfType::inverse, "computes inverse transformation" ).                                                                               \
        def( "__call__", &AffineXfType::operator(), "application of the transformation to a point" ).                                                              \
        def( pybind11::self* pybind11::self ).                                          \
        def( pybind11::self == pybind11::self );                                        \
} )

MR_ADD_PYTHON_AFFINE_XF( AffineXf2f )
MR_ADD_PYTHON_AFFINE_XF( AffineXf3f )
MR_ADD_PYTHON_AFFINE_XF( AffineXf2d )
MR_ADD_PYTHON_AFFINE_XF( AffineXf3d )

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, Line3f, MR::Line3f )
MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, Line3d, MR::Line3d )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Line, [] ( pybind11::module_& )
{
    MR_PYTHON_CUSTOM_CLASS( Line3f ).doc() =
        "3-dimensional line: cross( x - p, d ) = 0";
    MR_PYTHON_CUSTOM_CLASS( Line3f ).
        def( pybind11::init<>() ).
        def( pybind11::init<const MR::Vector3f&, const MR::Vector3f&>(), pybind11::arg( "p" ), pybind11::arg( "d" ) ).
        def_readwrite( "p", &MR::Line3f::p ).
        def_readwrite( "d", &MR::Line3f::d ).
        def( "distanceSq", &MR::Line3f::distanceSq, pybind11::arg( "x" ), "returns squared distance from given point to this line" ).
        def( "normalized", &MR::Line3f::normalized, "returns same line represented with unit d-vector" ).
        def( "project", &MR::Line3f::project, pybind11::arg( "x" ), "finds the closest point on line" );

    MR_PYTHON_CUSTOM_CLASS( Line3d ).doc() =
        "3-dimensional line: cross( x - p, d ) = 0";
    MR_PYTHON_CUSTOM_CLASS( Line3d ).
        def( pybind11::init<>() ).
        def( pybind11::init<const MR::Vector3d&, const MR::Vector3d&>(), pybind11::arg( "p" ), pybind11::arg( "d" ) ).
        def_readwrite( "p", &MR::Line3d::p ).
        def_readwrite( "d", &MR::Line3d::d ).
        def( "distanceSq", &MR::Line3d::distanceSq, pybind11::arg( "x" ), "returns squared distance from given point to this line" ).
        def( "normalized", &MR::Line3d::normalized, "returns same line represented with unit d-vector" ).
        def( "project", &MR::Line3d::project, pybind11::arg( "x" ), "finds the closest point on line" );
} )

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, Plane3f, MR::Plane3f )
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Plane3f, [] ( pybind11::module_& )
{
    MR_PYTHON_CUSTOM_CLASS( Plane3f ).doc() =
        "3-dimensional plane: dot(n,x) - d = 0";
    MR_PYTHON_CUSTOM_CLASS( Plane3f ).
        def( pybind11::init<>() ).
        def_readwrite( "n", &MR::Plane3f::n ).
        def_readwrite( "d", &MR::Plane3f::d );
} )

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, FaceId, MR::FaceId )
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, FaceId, [] ( pybind11::module_& )
{
    MR_PYTHON_CUSTOM_CLASS( FaceId ).
        def( pybind11::init<>() ).
        def( pybind11::init<int>() ).
        def( "valid", &MR::FaceId::valid ).
        def( "get", &MR::FaceId::operator int );
} )

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, VertId, MR::VertId )
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, VertId, [] ( pybind11::module_& )
{
    MR_PYTHON_CUSTOM_CLASS( VertId ).
        def( pybind11::init<>() ).
        def( pybind11::init<int>() ).
        def( "valid", &MR::VertId::valid ).
        def( "get", &MR::VertId::operator int );
} )

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, UndirectedEdgeId, MR::UndirectedEdgeId )
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, UndirectedEdgeId, [] ( pybind11::module_& )
{
    MR_PYTHON_CUSTOM_CLASS( UndirectedEdgeId ).
        def( pybind11::init<>() ).
        def( pybind11::init<int>() ).
        def( "valid", &MR::UndirectedEdgeId::valid ).
        def( "get", &MR::UndirectedEdgeId::operator int );
} )

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, ViewportId, MR::ViewportId )
MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, ViewportMask, MR::ViewportMask )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, ViewportId, [] ( pybind11::module_& )
{
    MR_PYTHON_CUSTOM_CLASS( ViewportId ).doc() =
        "stores unique identifier of a viewport, which is power of two;\n"
        "id=0 has a special meaning of default viewport in some contexts";
    MR_PYTHON_CUSTOM_CLASS( ViewportId ).
        def( pybind11::init<>() ).
        def( pybind11::init<unsigned>() ).
        def( "value", &MR::ViewportId::value ).
        def( "valid", &MR::ViewportId::valid );

    MR_PYTHON_CUSTOM_CLASS( ViewportMask ).doc() =
        "stores mask of viewport unique identifiers";
    MR_PYTHON_CUSTOM_CLASS( ViewportMask ).
        def( pybind11::init<>() ).
        def( pybind11::init<unsigned>() ).
        def( pybind11::init<MR::ViewportId>() ).
        def_static( "all", &MR::ViewportMask::all, "mask meaning all or any viewports" ).
        def_static( "any", &MR::ViewportMask::any, "mask meaning all or any viewports" );
} )

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, SegmPointf, MR::SegmPointf )
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, SegmPointf, [] ( pybind11::module_& )
{
    MR_PYTHON_CUSTOM_CLASS( SegmPointf ).
        def( pybind11::init<>() ).
        def( pybind11::init<float>(), pybind11::arg( "a" ) ).
        def_readwrite( "a", &MR::SegmPointf::a, "< a in [0,1], a=0 => point is in v0, a=1 => point is in v1" );
} )

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, EdgePoint, MR::EdgePoint )
MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, TriPointf, MR::TriPointf )
MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, MeshTriPoint, MR::MeshTriPoint )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshPoint, [] ( pybind11::module_& )
{
    MR_PYTHON_CUSTOM_CLASS( EdgePoint ).doc() =
        "encodes a point on a mesh edge";
    MR_PYTHON_CUSTOM_CLASS( EdgePoint ).
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

    MR_PYTHON_CUSTOM_CLASS( TriPointf ).doc() =
        "encodes a point inside a triangle using barycentric coordinates\n"
        "\tNotations used below: v0, v1, v2 - points of the triangle";
    MR_PYTHON_CUSTOM_CLASS( TriPointf ).
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
        def_readwrite( "b", &MR::TriPointf::b, "b in [0,1], b=0 => point is on [v0,v1] edge, b=1 => point is in v2" ).
        def( "interpolate", &MR::TriPointf::interpolate<float>, "given three values in three vertices, computes interpolated value at this barycentric coordinates" ).
        def( "interpolate", &MR::TriPointf::interpolate<MR::Vector2f>, "given three values in three vertices, computes interpolated value at this barycentric coordinates" ).
        def( "interpolate", &MR::TriPointf::interpolate<MR::Vector3f>, "given three values in three vertices, computes interpolated value at this barycentric coordinates" ); //Vector4f is not exposed to python yet

    MR_PYTHON_CUSTOM_CLASS( MeshTriPoint ).doc() =
        "encodes a point inside a triangular mesh face using barycentric coordinates\n"
        "\tNotations used below:\n"
        "\t v0 - the value in org( e )\n"
        "\t v1 - the value in dest( e )\n"
        "\t v2 - the value in dest( next( e ) )" ;
    MR_PYTHON_CUSTOM_CLASS( MeshTriPoint ).
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

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, EdgeId, MR::EdgeId )
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, EdgeId, [] ( pybind11::module_& )
{
    MR_PYTHON_CUSTOM_CLASS( EdgeId ).
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

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, BoostBitSet, boost::dynamic_bitset<uint64_t> )
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, BoostBitSet, [] ( pybind11::module_& )
{
    using type = boost::dynamic_bitset<uint64_t>;
    MR_PYTHON_CUSTOM_CLASS( BoostBitSet ).
        def( "size", &type::size ).
        def( "count", &type::count );
} )

#define ADD_PYTHON_BITSET( name, type ) \
MR_ADD_PYTHON_CUSTOM_CLASS_DECL( mrmeshpy, name, type, boost::dynamic_bitset<uint64_t> ) \
MR_ADD_PYTHON_CUSTOM_CLASS_INST( mrmeshpy, name )                                        \
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, name, [] ( pybind11::module_& )                          \
{\
    MR_PYTHON_CUSTOM_CLASS( name ).\
        def( pybind11::init<>() ).\
        def( "test", &type::test ).\
        def( "resize", &type::resize, pybind11::arg("size"), pybind11::arg("value") = false ).\
        def( "set",( type& ( type::* )( type::IndexType, bool ) )& type::set, pybind11::return_value_policy::reference, pybind11::arg("id"), pybind11::arg("value") = true ).\
        def( "flip",( type& ( type::* )() )& type::flip, pybind11::return_value_policy::reference ).\
        def( pybind11::self & pybind11::self ).\
        def( pybind11::self | pybind11::self ).\
        def( pybind11::self ^ pybind11::self ).\
        def( pybind11::self - pybind11::self ).\
        def( pybind11::self &= pybind11::self ).\
        def( pybind11::self |= pybind11::self ).\
        def( pybind11::self ^= pybind11::self ).\
        def( pybind11::self -= pybind11::self ).\
        def( "__iter__", [](type& data) {\
            return pybind11::make_iterator<pybind11::return_value_policy::copy>( begin( data ), end( data ) );\
        }, pybind11::keep_alive<0, 1>() );\
} )

ADD_PYTHON_BITSET( VertBitSet, MR::VertBitSet )
ADD_PYTHON_BITSET( UndirectedEdgeBitSet, MR::UndirectedEdgeBitSet )
ADD_PYTHON_BITSET( EdgeBitSet, MR::EdgeBitSet )
ADD_PYTHON_BITSET( FaceBitSet, MR::FaceBitSet )
ADD_PYTHON_BITSET( BitSet, MR::BitSet )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorVertBitSet, MR::VertBitSet )
MR_ADD_PYTHON_VEC( mrmeshpy, vectorFaceBitSet, MR::FaceBitSet )
MR_ADD_PYTHON_VEC( mrmeshpy, vectorUndirectedEdgeBitSet, MR::UndirectedEdgeBitSet )
