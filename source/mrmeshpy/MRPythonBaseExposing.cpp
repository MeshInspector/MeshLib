#include "MRMesh/MRPython.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRId.h"
#include "MRMesh/MRLine3.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRViewportId.h"

MR_INIT_PYTHON_MODULE( mrmeshpy )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Box3, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::Box3f>( m, "Box3" ).
        def( pybind11::init<>() ).
        def_readwrite( "min", &MR::Box3f::min ).
        def_readwrite( "max", &MR::Box3f::max ).
        def( "valid", &MR::Box3f::valid );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Vector2i, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::Vector2i>( m, "Vector2i" ).
        def( pybind11::init<>() ).
        def_readwrite( "x", &MR::Vector2i::x ).
        def_readwrite( "y", &MR::Vector2i::y ).
        def( pybind11::self + pybind11::self ).
        def( pybind11::self - pybind11::self ).
        def( pybind11::self * int() ).
        def( int() * pybind11::self ).
        def( pybind11::self / int() ).
        def( pybind11::self += pybind11::self ).
        def( pybind11::self -= pybind11::self ).
        def( pybind11::self *= int() ).
        def( pybind11::self /= int() ).
        def( -pybind11::self ).
        def( pybind11::self == pybind11::self ).
        def( "length", &MR::Vector2i::length ).
        def( "lengthSq", &MR::Vector2i::lengthSq ).
        def( "normalized", &MR::Vector2i::normalized );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Vector3, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::Vector3f>( m, "Vector3" ).
        def( pybind11::init<>() ).
        def( pybind11::init<float, float, float>() ).
        def_readwrite( "x", &MR::Vector3f::x ).
        def_readwrite( "y", &MR::Vector3f::y ).
        def_readwrite( "z", &MR::Vector3f::z ).
        def( pybind11::self + pybind11::self ).
        def( pybind11::self - pybind11::self ).
        def( pybind11::self* float() ).
        def( float()* pybind11::self ).
        def( pybind11::self / float() ).
        def( pybind11::self += pybind11::self ).
        def( pybind11::self -= pybind11::self ).
        def( pybind11::self *= float() ).
        def( pybind11::self /= float() ).
        def( -pybind11::self ).
        def( pybind11::self == pybind11::self ).
        def_static( "diagonal", &MR::Vector3f::diagonal ).
        def( "length", &MR::Vector3f::length ).
        def( "lengthSq", &MR::Vector3f::lengthSq ).
        def( "normalized", &MR::Vector3f::normalized );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Matrix3, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::Matrix3f>( m, "Matrix3" ).
        def( pybind11::init<>() ).
        def_readwrite( "x", &MR::Matrix3f::x ).
        def_readwrite( "y", &MR::Matrix3f::y ).
        def_readwrite( "z", &MR::Matrix3f::z ).
        def( "zero", &MR::Matrix3f::zero ).
        def( "scale", ( MR::Matrix3f( * )( float ) noexcept )& MR::Matrix3f::scale ).
        def( "scale", ( MR::Matrix3f( * )( float, float, float ) noexcept )& MR::Matrix3f::scale ).
        def( "rotation", ( MR::Matrix3f( * )( const MR::Vector3f&, float ) noexcept )& MR::Matrix3f::rotation ).
        def( "rotation", ( MR::Matrix3f( * )( const MR::Vector3f&, const MR::Vector3f& ) noexcept )& MR::Matrix3f::rotation ).
        def( "rotationFromEuler", &MR::Matrix3f::rotationFromEuler ).
        def( "normSq", &MR::Matrix3f::normSq ).
        def( "norm", &MR::Matrix3f::norm ).
        def( "det", &MR::Matrix3f::det ).
        def( "inverse", &MR::Matrix3f::inverse ).
        def( "transposed", &MR::Matrix3f::transposed ).
        def( "toEulerAngles", &MR::Matrix3f::toEulerAngles ).
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

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, AffineXf3, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::AffineXf3f>( m, "AffineXf3" ).
        def( pybind11::init<>() ).
        def_readwrite( "A", &MR::AffineXf3f::A ).
        def_readwrite( "b", &MR::AffineXf3f::b ).
        def( "translation", &MR::AffineXf3f::translation ).
        def( "linear", &MR::AffineXf3f::linear ).
        def( "xfAround", &MR::AffineXf3f::xfAround ).
        def( "linearOnly", &MR::AffineXf3f::linearOnly ).
        def( "inverse", &MR::AffineXf3f::inverse ).
        def( "__call__", &MR::AffineXf3f::operator() ).
        def( pybind11::self* pybind11::self ).
        def( pybind11::self == pybind11::self );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Line3, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::Line3f>( m, "Line3" ).
        def( pybind11::init<>() ).
        def_readwrite( "p", &MR::Line3f::p ).
        def_readwrite( "d", &MR::Line3f::d ).
        def( "distanceSq", &MR::Line3f::distanceSq ).
        def( "normalized", &MR::Line3f::normalized ).
        def( "project", &MR::Line3f::project );
} )

MR_ADD_PYTHON_FUNCTION( mrmeshpy, dot, ( float( * )( const MR::Vector3f&, const MR::Vector3f& ) )& MR::dot<float>, "dot product of two Vector3" )

MR_ADD_PYTHON_FUNCTION( mrmeshpy, cross, ( MR::Vector3f( * )( const MR::Vector3f&, const MR::Vector3f& ) )& MR::cross<float>, "cross product of two Vector3" )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Plane3, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::Plane3f>( m, "Plane3" ).
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
    pybind11::class_<MR::ViewportId>( m, "ViewportId" ).
        def( pybind11::init<>() ).
        def( pybind11::init<unsigned>() ).
        def( "value", &MR::ViewportId::value ).
        def( "valid", &MR::ViewportId::valid );

    pybind11::class_<MR::ViewportMask>( m, "ViewportMask" ).
        def( pybind11::init<>() ).
        def( pybind11::init<unsigned>() ).
        def( pybind11::init<MR::ViewportId>() ).
        def_static( "all", &MR::ViewportMask::all ).
        def_static( "any", &MR::ViewportMask::any );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, EdgeId, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::EdgeId>( m, "EdgeId" ).
        def( pybind11::init<>() ).
        def( pybind11::init<int>() ).
        def( "valid", &MR::EdgeId::valid ).
        def( "sym", &MR::EdgeId::sym ).
        def( "undirected", &MR::EdgeId::undirected ).
        def( "get", &MR::EdgeId::operator int );
} )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorEdges, MR::EdgeId )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorVerts, MR::VertId )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorFaces, MR::FaceId )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorVec3, MR::Vector3f )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, BoostBitSet, [] ( pybind11::module_& m )
{
    using type = boost::dynamic_bitset<uint64_t>;
    pybind11::class_<type>( m, "BoostBitSet" ).
        def( "size", &type::size ).
        def( "flip", ( type& ( type::* )( ) )& type::flip, pybind11::return_value_policy::reference ).
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
