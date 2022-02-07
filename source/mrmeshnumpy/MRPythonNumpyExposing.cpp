#include "MRMesh/MREmbeddedPython.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshBuilder.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRId.h"
#include "MRMesh/MRPointCloud.h"

MR_INIT_PYTHON_MODULE_PRECALL( mrmeshnumpy, [] ()
{
    pybind11::module_::import( "mrmeshpy" );
} )


MR::Mesh fromFV( const pybind11::buffer& faces, const pybind11::buffer& verts )
{
    pybind11::buffer_info infoFaces = faces.request();
    pybind11::buffer_info infoVerts = verts.request();
    if ( infoFaces.ndim != 2 || infoFaces.shape[1] != 3 )
    {
        PyErr_SetString( PyExc_RuntimeError, "shape of input python vector 'faces' should be (n,3)" );
        assert( false );
    }
    if ( infoVerts.ndim != 2 || infoVerts.shape[1] != 3 )
    {
        PyErr_SetString( PyExc_RuntimeError, "shape of input python vector 'verts' should be (n,3)" );
        assert( false );
    }

    MR::Mesh res;

    // faces to topology part
    std::vector<MR::MeshBuilder::Triangle> triangles( infoFaces.shape[0] );
    if ( infoFaces.itemsize == sizeof( int ) )
    {
        int* data = reinterpret_cast< int* >( infoFaces.ptr );
        for ( auto i = 0; i < infoFaces.shape[0]; i++ )
        {
            triangles[i] = MR::MeshBuilder::Triangle( MR::VertId( data[3 * i] ), MR::VertId( data[3 * i + 1] ), MR::VertId( data[3 * i + 2] ), MR::FaceId( i ) );
        }
    }
    else
    {
        // format of input python vector is not numeric
        PyErr_SetString( PyExc_RuntimeError, "dtype of input python vector 'faces' should be int32" );
        assert( false );
    }
    res.topology = MR::MeshBuilder::fromTriangles( triangles );

    // verts to points part
    res.points.resize( infoVerts.shape[0] );
    if ( infoVerts.format == pybind11::format_descriptor<double>::format() )
    {
        double* data = reinterpret_cast< double* >( infoVerts.ptr );
        for ( auto i = 0; i < infoVerts.shape[0]; i++ )
        {
            res.points[MR::VertId( i )] = MR::Vector3f( float( data[3 * i] ), float( data[3 * i + 1] ), float( data[3 * i + 2] ) );
        }
    }
    else if ( infoVerts.format == pybind11::format_descriptor<float>::format() )
    {
        float* data = reinterpret_cast< float* >( infoVerts.ptr );
        for ( auto i = 0; i < infoVerts.shape[0]; i++ )
        {
            res.points[MR::VertId( i )] = MR::Vector3f( data[3 * i], data[3 * i + 1], data[3 * i + 2] );
        }
    }
    else
    {
        PyErr_SetString( PyExc_RuntimeError, "dtype of input python vector 'verts' should be float32 or float64" );
        assert( false );
    }

    return res;
}

MR_ADD_PYTHON_FUNCTION( mrmeshnumpy, topologyFromFacesVerts, &fromFV, "constructs topology from given numpy ndarrays of faces (N VertId x3, FaceId x1), verts (M vec3 x3)" )


MR::PointCloud pointCloudFromNP( const pybind11::buffer& points, const pybind11::buffer& normals )
{
    pybind11::buffer_info infoPoints = points.request();
    pybind11::buffer_info infoNormals = normals.request();
    if ( infoPoints.ndim != 2 || infoPoints.shape[1] != 3 )
    {
        PyErr_SetString( PyExc_RuntimeError, "shape of input python vector 'points' should be (n,3)" );
        assert( false );
    }
    if ( infoNormals.size != 0 && ( infoNormals.ndim != 2 || infoNormals.shape[1] != 3 ) )
    {
        PyErr_SetString( PyExc_RuntimeError, "shape of input python vector 'normals' should be (n,3) or empty" );
        assert( false );
    }

    MR::PointCloud res;

    auto fillFloatVec = [] ( MR::VertCoords& vec, const pybind11::buffer_info& bufInfo )
    {
        vec.resize( bufInfo.shape[0] );
        if ( bufInfo.format == pybind11::format_descriptor<double>::format() )
        {
            double* data = reinterpret_cast< double* >( bufInfo.ptr );
            for ( auto i = 0; i < bufInfo.shape[0]; i++ )
            {
                vec[MR::VertId( i )] = MR::Vector3f( float( data[3 * i] ), float( data[3 * i + 1] ), float( data[3 * i + 2] ) );
            }
        }
        else if ( bufInfo.format == pybind11::format_descriptor<float>::format() )
        {
            float* data = reinterpret_cast< float* >( bufInfo.ptr );
            for ( auto i = 0; i < bufInfo.shape[0]; i++ )
            {
                vec[MR::VertId( i )] = MR::Vector3f( data[3 * i], data[3 * i + 1], data[3 * i + 2] );
            }
        }
        else
        {
            PyErr_SetString( PyExc_RuntimeError, "dtype of input python vector should be float32 or float64" );
            assert( false );
        }
    };

    // verts to points part
    fillFloatVec( res.points, infoPoints );
    if ( infoNormals.size > 0 )
        fillFloatVec( res.normals, infoNormals );

    res.validPoints = MR::VertBitSet( res.points.size() );
    res.validPoints.flip();

    return res;
}

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshnumpy, PointCloudFromPoints, [] ( pybind11::module_& m )
{
    m.def( "pointCloudFromPoints", &pointCloudFromNP, pybind11::arg( "points" ), pybind11::arg( "normals" ) = pybind11::array{}, "creates point cloud object from numpy arrays, first arg - points, second optional arg - normals" );
} )
