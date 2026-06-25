#include "MRPythonNumpy.h"
#include "MRPython/MRPython.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshBuilder.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRId.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRCloseVertices.h"
#include "MRMesh/MRParallelFor.h"
#include "MRMesh/MRMacros.h"

MR_INIT_PYTHON_MODULE_PRECALL( mrmeshnumpy, [] ()
{
    pybind11::module_::import( MR_STR( MRMESHNUMPY_PARENT_MODULE_NAME ) ".mrmeshpy" );
} )

std::vector<MR::Vector3f> fromNumpyArrayInfo( const pybind11::buffer_info& bufInfo )
{
    std::vector<MR::Vector3f> vec;
    auto stride0 = bufInfo.strides[0] / bufInfo.itemsize;
    auto stride1 = bufInfo.strides[1] / bufInfo.itemsize;
    vec.resize( bufInfo.shape[0] );
    auto fillData = [&] ( const auto* data )
    {
        for ( auto i = 0; i < bufInfo.shape[0]; i++ )
        {
            auto ind = stride0 * i;
            vec[i] = MR::Vector3f(
                float( data[ind] ),
                float( data[ind + stride1] ),
                float( data[ind + stride1 * 2] ) );
        }
    };
    if ( bufInfo.format == pybind11::format_descriptor<double>::format() )
    {
        double* data = reinterpret_cast< double* >( bufInfo.ptr );
        fillData( data );
    }
    else if ( bufInfo.format == pybind11::format_descriptor<float>::format() )
    {
        float* data = reinterpret_cast< float* >( bufInfo.ptr );
        fillData( data );
    }
    else
        throw std::runtime_error( "dtype of input python vector should be float32 or float64" );
    return vec;
}

std::vector<MR::Vector3f> fromNumpyArray( const pybind11::buffer& coords )
{
    pybind11::buffer_info bufInfo = coords.request();
    if ( bufInfo.ndim != 2 || bufInfo.shape[1] != 3 )
        throw std::runtime_error( "shape of input python vector 'coords' should be (n,3)" );

    return fromNumpyArrayInfo( bufInfo );
}

MR::Mesh fromFV( const pybind11::buffer& faces, const pybind11::buffer& verts, const MR::MeshBuilder::BuildSettings & settings, bool duplicateNonManifoldVertices )
{
    pybind11::buffer_info infoFaces = faces.request();
    pybind11::buffer_info infoVerts = verts.request();
    if ( infoFaces.ndim != 2 || infoFaces.shape[1] != 3 )
        throw std::runtime_error( "shape of input python vector 'faces' should be (n,3)" );
    if ( infoVerts.ndim != 2 || infoVerts.shape[1] != 3 )
        throw std::runtime_error( "shape of input python vector 'verts' should be (n,3)" );

    // faces to topology part
    auto strideF0 = infoFaces.strides[0] / infoFaces.itemsize;
    auto strideF1 = infoFaces.strides[1] / infoFaces.itemsize;
    MR::Triangulation t;

    auto fillTris = [&] ( const auto* data )
    {
        t.reserve( infoFaces.shape[0] );
        for ( auto i = 0; i < infoFaces.shape[0]; i++ )
        {
            auto ind = strideF0 * i;
            t.push_back( {
                MR::VertId( int( data[ind] ) ),
                MR::VertId( int( data[ind + strideF1] ) ),
                MR::VertId( int( data[ind + strideF1 * 2] ) )
                } );
        }
    };
    if ( infoFaces.itemsize == sizeof( int32_t ) )
    {
        int* data = reinterpret_cast< int32_t* >( infoFaces.ptr );
        fillTris( data );
    }
    else if ( infoFaces.itemsize == sizeof( int64_t ) )
    {
        int64_t* data = reinterpret_cast< int64_t* >( infoFaces.ptr );
        fillTris( data );
    }
    else
        throw std::runtime_error( "dtype of input python vector 'faces' should be int32 or int64" );

    // verts to points part
    MR::VertCoords points;
    points.vec_ = fromNumpyArrayInfo( infoVerts );
    if ( duplicateNonManifoldVertices )
        return MR::Mesh::fromTrianglesDuplicatingNonManifoldVertices( std::move( points ), t, nullptr, settings );
    else
        return MR::Mesh::fromTriangles( std::move( points ), t, settings );
}

MR::Mesh fromUVPoints( const pybind11::buffer& xArray, const pybind11::buffer& yArray, const pybind11::buffer& zArray )
{
    pybind11::buffer_info xInfo = xArray.request();
    pybind11::buffer_info yInfo = yArray.request();
    pybind11::buffer_info zInfo = zArray.request();

    MR::Vector2i shape;
    int format = -1; // 0 - float, 1 - double
    auto checkArray = [&] ( const pybind11::buffer_info& info, const std::string& arrayName )->bool
    {
        if ( info.ndim != 2 )
        {
            std::string error = arrayName + " should be 2D";
            throw std::runtime_error( error.c_str() );
        }
        MR::Vector2i thisShape;
        thisShape.x = int( info.shape[0] );
        thisShape.y = int( info.shape[1] );
        if ( shape == MR::Vector2i() )
            shape = thisShape;
        else if ( shape != thisShape )
        {
            std::string error = "Input arrays shapes should be same";
            throw std::runtime_error( error.c_str() );
        }
        int thisFormat = -1;
        if ( info.format == pybind11::format_descriptor<float>::format() )
            thisFormat = 0;
        else if ( info.format == pybind11::format_descriptor<double>::format() )
            thisFormat = 1;

        if ( format == -1 )
            format = thisFormat;

        if ( format == -1 )
        {
            std::string error = arrayName + " dtype should be float32 or float64";
            throw std::runtime_error( error.c_str() );
        }
        if ( format != thisFormat )
        {
            std::string error = "Arrays should have same dtype";
            throw std::runtime_error( error.c_str() );
        }
        return true;
    };

    if ( !checkArray( xInfo, "X" ) || !checkArray( yInfo, "Y" ) || !checkArray( zInfo, "Z" ) )
        return {};

    assert( format != -1 );
    float* fDataPtr[3];
    double* dDataPtr[3];
    std::function<float( int coord, int i )> getter;
    if ( format == 0 )
    {
        fDataPtr[0] = reinterpret_cast< float* >( xInfo.ptr );
        fDataPtr[1] = reinterpret_cast< float* >( yInfo.ptr );
        fDataPtr[2] = reinterpret_cast< float* >( zInfo.ptr );
        getter = [&] ( int coord, int i )->float
        {
            int u = i % shape.x;
            int v = i / shape.x;
            int ind = u * shape.y + v;
            return fDataPtr[coord][ind];
        };
    }
    else
    {
        dDataPtr[0] = reinterpret_cast< double* >( xInfo.ptr );
        dDataPtr[1] = reinterpret_cast< double* >( yInfo.ptr );
        dDataPtr[2] = reinterpret_cast< double* >( zInfo.ptr );
        getter = [&] ( int coord, int i )->float
        {
            int u = i % shape.x;
            int v = i / shape.x;
            int ind = u * shape.y + v;
            return float( dDataPtr[coord][ind] );
        };
    }

    MR::Mesh res;
    res.points.resize( shape.x * shape.y );
    ParallelFor( res.points, [&]( MR::VertId i )
    {
        res.points[i] = MR::Vector3f( getter( 0, i ), getter( 1, i ), getter( 2, i ) );
    } );

    int triangleCount = 2 * int( res.points.size() ) - 2 * shape.x;
    MR::Triangulation t;
    t.reserve( triangleCount );

    for ( int i = 0; i < shape.y; ++i )
    {
        for ( int j = 0; j < shape.x; ++j )
        {
            if ( i + 1 < shape.y && j + 1 < shape.x )
            {
                t.push_back( {
                    MR::VertId( i * shape.x + j ),
                    MR::VertId( i * shape.x + ( j + 1 ) ),
                    MR::VertId( ( i + 1 ) * shape.x + j ) } );
            }
            if ( i > 0 && j > 0 )
            {
                t.push_back( {
                    MR::VertId( i * shape.x + j ),
                    MR::VertId( i * shape.x + ( j - 1 ) ),
                    MR::VertId( ( i - 1 ) * shape.x + j ) } );
            }
        }
    }

    // unite close vertices
    const auto vertOldToNew = *findSmallestCloseVertices( res.points, std::numeric_limits<float>::epsilon() );
    for ( auto & tri : t )
    {
        for ( int i = 0; i < 3; ++i )
            tri[i] = vertOldToNew[tri[i]];
    }

    res.topology = MR::MeshBuilder::fromTriangles( t );

    if ( res.volume() < 0.0f )
        res.topology.flipOrientation();

    res.pack();

    return res;
}

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshnumpy, NumpyMesh, [] ( pybind11::module_& m )
{
    m.def( "meshFromFacesVerts", &fromFV,
    pybind11::arg( "faces" ), pybind11::arg( "verts" ), pybind11::arg_v( "settings", MR::MeshBuilder::BuildSettings(), "MeshBuilderSettings()" ),
    pybind11::arg( "duplicateNonManifoldVertices" ) = true,
            "constructs mesh from given numpy ndarrays of faces (N VertId x3, FaceId x1), verts (M vec3 x3)" );
    m.def( "meshFromUVPoints", &fromUVPoints, pybind11::arg( "xArray" ), pybind11::arg( "yArray" ), pybind11::arg( "zArray" ),
            "constructs mesh from three 2d numpy ndarrays with x,y,z positions of mesh" );
} )

MR::PointCloud pointCloudFromNP( const pybind11::buffer& points, const pybind11::buffer& normals )
{
    pybind11::buffer_info infoPoints = points.request();
    pybind11::buffer_info infoNormals = normals.request();
    if ( infoPoints.ndim != 2 || infoPoints.shape[1] != 3 )
        throw std::runtime_error( "shape of input python vector 'points' should be (n,3)" );
    if ( infoNormals.size != 0 && ( infoNormals.ndim != 2 || infoNormals.shape[1] != 3 ) )
        throw std::runtime_error( "shape of input python vector 'normals' should be (n,3) or empty" );

    MR::PointCloud res;

    // verts to points part
    res.points = fromNumpyArrayInfo( infoPoints );
    if ( infoNormals.size > 0 )
        res.normals = fromNumpyArrayInfo( infoNormals );

    res.validPoints = MR::VertBitSet( res.points.size() );
    res.validPoints.flip();

    return res;
}

MR::Polyline2 polyline2FromNP( const pybind11::buffer& points )
{
    pybind11::buffer_info infoPoints = points.request();
    if ( infoPoints.ndim != 2 || infoPoints.shape[1] != 2 )
        throw std::runtime_error( "shape of input python vector 'points' should be (n,2)" );

    // verts to points part
    MR::Contour2f inputContour;

    auto stride0 = infoPoints.strides[0] / infoPoints.itemsize;
    auto stride1 = infoPoints.strides[1] / infoPoints.itemsize;
    inputContour.resize( infoPoints.shape[0] );
    auto fillPoints = [&] ( const auto* data )
    {
        for ( auto i = 0; i < infoPoints.shape[0]; i++ )
        {
            auto ind = stride0 * i;
            inputContour[i] = MR::Vector2f( float( data[ind] ), float( data[ind + stride1] ) );
        }
    };
    if ( infoPoints.format == pybind11::format_descriptor<double>::format() )
    {
        double* data = reinterpret_cast< double* >( infoPoints.ptr );
        fillPoints( data );
    }
    else if ( infoPoints.format == pybind11::format_descriptor<float>::format() )
    {
        float* data = reinterpret_cast< float* >( infoPoints.ptr );
        fillPoints( data );
    }
    else
        throw std::runtime_error( "dtype of input python vector should be float32 or float64" );

    MR::Polyline2 res;
    res.addFromPoints( inputContour.data(), inputContour.size() );

    return res;
}

// returns numpy array shapes [num faces,3] which represents vertices of mesh valid faces
pybind11::array_t<int> getNumpyFaces( const MR::MeshTopology& topology )
{
    using namespace MR;
    const auto& validFaces = topology.getValidFaces();
    int numFaces = topology.lastValidFace() + 1;
    // Allocate and initialize some data;
    const int size = numFaces * 3;
    int* data = new int[size];
    ParallelFor( 0_f, FaceId( numFaces), [&]( FaceId f )
    {
        int ind = 3 * f;
        if ( validFaces.test( f ) )
        {
            VertId v[3];
            topology.getTriVerts( f, v );
            for ( int vi = 0; vi < 3; ++vi )
                data[ind + vi] = v[vi];
        }
        else
        {
            for ( int vi = 0; vi < 3; ++vi )
                data[ind + vi] = 0;
        }
    } );

    // Create a Python object that will free the allocated
    // memory when destroyed:
    pybind11::capsule freeWhenDone( data, [] ( void* f )
    {
        int* data = reinterpret_cast< int* >( f );
        delete[] data;
    } );

    return pybind11::array_t<int>(
        { numFaces, 3 }, // shape
        { 3 * sizeof( int ), sizeof( int ) }, // C-style contiguous strides for int
        data, // the data pointer
        freeWhenDone ); // numpy array references this parent
}

// returns numpy array shapes [num verts,3] which represents coordinates from given vector
pybind11::array_t<double> toNumpyArray( const std::vector<MR::Vector3f>& coords )
{
    using namespace MR;
    auto numVerts = coords.size();
    // Allocate and initialize some data;
    const auto size = numVerts * 3;
    double* data = new double[size];
    ParallelFor( coords, [&]( size_t i )
    {
        auto ind = 3 * i;
        for ( int vi = 0; vi < 3; ++vi )
            data[ind + vi] = coords[i][vi];
    } );

    // Create a Python object that will free the allocated
    // memory when destroyed:
    pybind11::capsule freeWhenDone( data, [] ( void* f )
    {
        double* data = reinterpret_cast< double* >( f );
        delete[] data;
    } );

    return pybind11::array_t<double>(
        { (int)numVerts, 3 }, // shape
        { 3 * sizeof( double ), sizeof( double ) }, // C-style contiguous strides for double
        data, // the data pointer
        freeWhenDone ); // numpy array references this parent
}

pybind11::array_t<double> toNumpyArray( const MR::VertCoords& coords )
{
    return toNumpyArray( coords.vec_ );
}

pybind11::array_t<double> toNumpyArray( const MR::FaceNormals& norms )
{
    return toNumpyArray( norms.vec_ );
}

// returns numpy array shapes [num verts,3] which represents coordinates of all mesh points (including invalid ones)
pybind11::array_t<double> getNumpyVerts( const MR::Mesh& mesh )
{
    return toNumpyArray( mesh.points );
}

pybind11::array_t<bool> getNumpyBitSet( const MR::BitSet& bitSet )
{
    using namespace MR;
    // Allocate and initialize some data;
    const size_t size = bitSet.size();
    bool* data = new bool[size];
    for ( int i = 0; i < size; i++ )
    {
        data[i] = bitSet.test( i );
    }

    // Create a Python object that will free the allocated
    // memory when destroyed:
    pybind11::capsule freeWhenDone( data, [] ( void* f )
    {
        bool* data = reinterpret_cast< bool* >( f );
        delete[] data;
    } );

    return pybind11::array_t<bool>(
        { size }, // shape
        { sizeof( bool ) }, // C-style contiguous strides for bool
        data, // the data pointer
        freeWhenDone ); // numpy array references this parent
}

pybind11::array_t<double> getNumpyMeanCurvature( const MR::Mesh& mesh )
{
    using namespace MR;
    // Allocate and initialize some data;
    int numVerts = mesh.topology.lastValidVert() + 1;
    double* data = new double[numVerts];
    ParallelFor( 0_v, VertId( numVerts ), [&]( VertId v )
    {
        if ( mesh.topology.hasVert( v ) )
            data[v] = double( mesh.discreteMeanCurvature( v ) );
        else
            data[v] = 0.0;
    } );

    // Create a Python object that will free the allocated
    // memory when destroyed:
    pybind11::capsule freeWhenDone( data, [] ( void* f )
    {
        double* data = reinterpret_cast< double* >( f );
        delete[] data;
    } );

    return pybind11::array_t<double>(
        { numVerts }, // shape
        { sizeof( double ) }, // C-style contiguous strides for double
        data, // the data pointer
        freeWhenDone ); // numpy array references this parent
}

pybind11::array_t<double> getNumpyGaussianCurvature( const MR::Mesh& mesh )
{
    using namespace MR;
    // Allocate and initialize some data;
    int numVerts = mesh.topology.lastValidVert() + 1;
    double* data = new double[numVerts];
    ParallelFor( 0_v, VertId( numVerts ), [&]( VertId v )
    {
        if ( mesh.topology.hasVert( v ) )
            data[v] = double( mesh.discreteGaussianCurvature( v ) );
        else
            data[v] = 0.0;
    } );

    // Create a Python object that will free the allocated
    // memory when destroyed:
    pybind11::capsule freeWhenDone( data, [] ( void* f )
    {
        double* data = reinterpret_cast< double* >( f );
        delete[] data;
    } );

    return pybind11::array_t<double>(
        { numVerts }, // shape
        { sizeof( double ) }, // C-style contiguous strides for double
        data, // the data pointer
        freeWhenDone ); // numpy array references this parent
}

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshnumpy, NumpyMeshData, [] ( pybind11::module_& m )
{
    m.def( "getNumpyCurvature", &getNumpyMeanCurvature, pybind11::arg( "mesh" ), "retunrs numpy array with discrete mean curvature for each vertex of a mesh" ); // to be deprecated
    m.def( "getNumpyMeanCurvature", &getNumpyMeanCurvature, pybind11::arg( "mesh" ), "retunrs numpy array with discrete mean curvature for each vertex of a mesh" );
    m.def( "getNumpyGaussianCurvature", &getNumpyGaussianCurvature, pybind11::arg( "mesh" ), "retunrs numpy array with discrete Gaussian curvature for each vertex of a mesh" );
    m.def( "getNumpyFaces", &getNumpyFaces, pybind11::arg( "topology" ), "returns numpy array shapes [num faces,3] which represents vertices of mesh valid faces " );
    m.def( "getNumpyVerts", &getNumpyVerts, pybind11::arg( "mesh" ), "returns numpy array shapes [num verts,3] which represents coordinates of all mesh points (including invalid ones)" );
    m.def( "getNumpyBitSet", &getNumpyBitSet, pybind11::arg( "bitset" ), "returns numpy array with bools for each bit of given bitset" );
    m.def( "toNumpyArray", ( pybind11::array_t<double>( * )( const MR::VertCoords& ) )& toNumpyArray, pybind11::arg( "coords" ), "returns numpy array shapes [num coords,3] which represents coordinates from given vector" );
    m.def( "toNumpyArray", ( pybind11::array_t<double>( * )( const MR::FaceNormals& ) )& toNumpyArray, pybind11::arg( "coords" ), "returns numpy array shapes [num coords,3] which represents coordinates from given vector" );
    m.def( "toNumpyArray", ( pybind11::array_t<double>( * )( const std::vector<MR::Vector3f>& ) )& toNumpyArray, pybind11::arg( "coords" ), "returns numpy array shapes [num coords,3] which represents coordinates from given vector" );
    m.def( "fromNumpyArray", &fromNumpyArray, pybind11::arg( "coords" ), "constructs mrmeshpy.vectorVector3f from numpy ndarray with shape (n,3)" );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshnumpy, PointCloudFromPoints, [] ( pybind11::module_& m )
{
    m.def( "pointCloudFromPoints", &pointCloudFromNP,
        pybind11::arg( "points" ), pybind11::arg( "normals" ) = pybind11::array{},
        "creates point cloud object from numpy arrays, first arg - points, second optional arg - normals" );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshnumpy, Polyline2FromPoints, [] ( pybind11::module_& m )
{
    m.def( "polyline2FromPoints", &polyline2FromNP,
        pybind11::arg( "points" ),
        "creates polyline2 object from numpy array" );
} )

template<typename T>
MR::TaggedBitSet<T> bitSetFromNP( const pybind11::buffer& bools )
{
    pybind11::buffer_info boolsInfo = bools.request();
    if ( boolsInfo.ndim != 1 )
        throw std::runtime_error( "shape of input python vector 'bools' should be (n)" );

    if ( boolsInfo.shape[0] == 0 )
        return {};

    if ( boolsInfo.format != pybind11::format_descriptor<bool>::format() )
        throw std::runtime_error( "format of python vector 'bools' should be bool" );

    MR::TaggedBitSet<T> resultBitSet( boolsInfo.shape[0] );

    bool* data = reinterpret_cast< bool* >( boolsInfo.ptr );
    for ( int i = 0; i < boolsInfo.shape[0]; ++i )
        resultBitSet.set( MR::Id<T>( i ), data[i] );

    return resultBitSet;
}

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshnumpy, NumpyBitSets, [] ( pybind11::module_& m )
{
    m.def( "faceBitSetFromBools", &bitSetFromNP<MR::FaceTag>, pybind11::arg( "boolArray" ), "returns FaceBitSet from numpy array with bools" );
    m.def( "vertBitSetFromBools", &bitSetFromNP<MR::VertTag>, pybind11::arg( "boolArray" ), "returns VertBitSet from numpy array with bools" );
    m.def( "edgeBitSetFromBools", &bitSetFromNP<MR::EdgeTag>, pybind11::arg( "boolArray" ), "returns EdgeBitSet from numpy array with bools" );
    m.def( "undirectedEdgeBitSetFromBools", &bitSetFromNP<MR::UndirectedEdgeTag>, pybind11::arg( "boolArray" ), "returns UndirectedEdgeBitSet from numpy array with bools" );
} )
