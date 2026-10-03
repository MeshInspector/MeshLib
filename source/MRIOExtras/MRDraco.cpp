#include "MRDraco.h"
#ifndef MRIOEXTRAS_NO_DRACO

#include <MRMesh/MRColor.h>
#include <MRMesh/MRIOFormatsRegistry.h>
#include <MRMesh/MRIOParsing.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshBuilder.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRProgressReadWrite.h>
#include <MRMesh/MRStringConvert.h>
#include <MRMesh/MRTimer.h>
#include <MRMesh/MRphmap.h>
#include <MRPch/MRSuppressWarning.h>

MR_SUPPRESS_WARNING_PUSH
#pragma warning( disable : 4100 ) // unreferenced formal parameter
#pragma warning( disable : 4127 ) // conditional expression is constant
#pragma warning( disable : 4244 ) // conversion, possible loss of data
#pragma warning( disable : 4267 ) // conversion from 'size_t', possible loss of data
#if defined( __GNUC__ ) || defined( __clang__ )
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <draco/compression/decode.h>
#include <draco/compression/encode.h>
MR_SUPPRESS_WARNING_POP

#include <fstream>

namespace MR
{

namespace
{

Vector3f getVector3( const draco::PointAttribute& att, draco::PointIndex p )
{
    Vector3f res;
    att.ConvertValue<float, 3>( att.mapped_index( p ), &res.x );
    return res;
}

UVCoord getUV( const draco::PointAttribute& att, draco::PointIndex p )
{
    UVCoord res;
    att.ConvertValue<float, 2>( att.mapped_index( p ), &res.x );
    return res;
}

Color getColor( const draco::PointAttribute& att, draco::PointIndex p )
{
    const auto numComponents = int8_t( std::min( int( att.num_components() ), 4 ) );
    if ( att.data_type() == draco::DT_UINT8 )
    {
        uint8_t c[4] = { 0, 0, 0, 255 };
        att.ConvertValue<uint8_t>( att.mapped_index( p ), numComponents, c );
        return Color( c[0], c[1], c[2], c[3] );
    }
    // other types are converted in [0,1] range if normalized
    float c[4] = { 0, 0, 0, 1 };
    att.ConvertValue<float>( att.mapped_index( p ), numComponents, c );
    return Color( c[0], c[1], c[2], c[3] );
}

struct DecodedGeometry
{
    std::unique_ptr<draco::PointCloud> cloud;
    const draco::Mesh* mesh = nullptr; // not null if the file contains a mesh, points to (cloud)
};

Expected<DecodedGeometry> decode( std::istream& in, const ProgressCallback& cb )
{
    auto data = readCharBuffer( in );
    if ( !data )
        return unexpected( std::move( data.error() ) );
    if ( !reportProgress( cb, 0.25f ) )
        return unexpectedOperationCanceled();

    draco::DecoderBuffer buffer;
    buffer.Init( data->data(), data->size() );
    auto type = draco::Decoder::GetEncodedGeometryType( &buffer );
    if ( !type.ok() )
        return unexpected( "Draco decoding error: " + type.status().error_msg_string() );

    draco::Decoder decoder;
    auto res = decoder.DecodePointCloudFromBuffer( &buffer );
    if ( !res.ok() )
        return unexpected( "Draco decoding error: " + res.status().error_msg_string() );
    if ( !reportProgress( cb, 0.5f ) )
        return unexpectedOperationCanceled();

    DecodedGeometry geom;
    geom.cloud = std::move( res ).value();
    if ( type.value() == draco::TRIANGULAR_MESH )
        geom.mesh = static_cast<const draco::Mesh*>( geom.cloud.get() );
    if ( !geom.cloud->GetNamedAttribute( draco::GeometryAttribute::POSITION ) )
        return unexpected( "Draco file has no vertex positions" );
    return geom;
}

/// adds new attribute in (pc) with the values of given type (V) for all (verts) in order
template <typename V, typename F>
void addAttribute( draco::PointCloud& pc, draco::GeometryAttribute::Type type, int numComponents, draco::DataType dataType,
    const std::vector<VertId>& verts, F&& getValue )
{
    draco::GeometryAttribute ga;
    ga.Init( type, nullptr, uint8_t( numComponents ), dataType, dataType == draco::DT_UINT8, sizeof( V ), 0 );
    auto& att = *pc.attribute( pc.AddAttribute( ga, true, uint32_t( verts.size() ) ) );
    for ( size_t i = 0; i < verts.size(); ++i )
    {
        const V value = getValue( verts[i] );
        att.SetAttributeValue( draco::AttributeValueIndex( uint32_t( i ) ), &value );
    }
}

Expected<void> encode( const draco::PointCloud& pc, const draco::Mesh* mesh, std::ostream& out, const DracoSaveOptions& options )
{
    draco::Encoder encoder;
    const int speed = 10 - std::clamp( options.compressionLevel, 0, 10 );
    encoder.SetSpeedOptions( speed, speed );
    if ( options.positionQuantizationBits > 0 )
        encoder.SetAttributeQuantization( draco::GeometryAttribute::POSITION, options.positionQuantizationBits );

    draco::EncoderBuffer buffer;
    const auto status = mesh ? encoder.EncodeMeshToBuffer( *mesh, &buffer ) : encoder.EncodePointCloudToBuffer( pc, &buffer );
    if ( !status.ok() )
        return unexpected( "Draco encoding error: " + status.error_msg_string() );
    if ( !reportProgress( options.progress, 0.5f ) )
        return unexpectedOperationCanceled();

    if ( !writeByBlocks( out, buffer.data(), buffer.size(), subprogress( options.progress, 0.5f, 1.0f ) ) )
        return unexpectedOperationCanceled();
    if ( !out )
        return unexpected( std::string( "Error writing Draco data in stream" ) );
    return {};
}

} // anonymous namespace

namespace MeshLoad
{

Expected<Mesh> fromDrc( const std::filesystem::path& file, const MeshLoadSettings& settings )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromDrc( in, settings ), file );
}

Expected<Mesh> fromDrc( std::istream& in, const MeshLoadSettings& settings )
{
    MR_TIMER;
    auto geom = decode( in, settings.callback );
    if ( !geom )
        return unexpected( std::move( geom.error() ) );
    const auto& pc = *geom->cloud;

    const auto& posAtt = *pc.GetNamedAttribute( draco::GeometryAttribute::POSITION );
    const auto* colorAtt = settings.colors ? pc.GetNamedAttribute( draco::GeometryAttribute::COLOR ) : nullptr;
    const auto* normalAtt = settings.normals ? pc.GetNamedAttribute( draco::GeometryAttribute::NORMAL ) : nullptr;
    const auto* uvAtt = settings.uvCoords ? pc.GetNamedAttribute( draco::GeometryAttribute::TEX_COORD ) : nullptr;

    // Draco splits the vertices on attribute seams, so merge them back by coordinates in meshes
    const auto numPoints = pc.num_points();
    std::vector<VertId> pointToVert( numPoints );
    HashMap<Vector3f, VertId> posToVert;
    VertCoords points;
    points.reserve( numPoints );
    VertColors colors;
    VertNormals normals;
    VertUVCoords uvCoords;
    for ( draco::PointIndex p( 0 ); p < numPoints; ++p )
    {
        const auto pos = getVector3( posAtt, p );
        if ( geom->mesh )
        {
            auto [it, inserted] = posToVert.insert( { pos, points.endId() } );
            pointToVert[p.value()] = it->second;
            if ( !inserted )
                continue;
        }
        else
            pointToVert[p.value()] = points.endId();

        points.push_back( pos );
        if ( colorAtt )
            colors.push_back( getColor( *colorAtt, p ) );
        if ( normalAtt )
            normals.push_back( getVector3( *normalAtt, p ) );
        if ( uvAtt )
            uvCoords.push_back( getUV( *uvAtt, p ) );
    }
    if ( !reportProgress( settings.callback, 0.75f ) )
        return unexpectedOperationCanceled();

    Triangulation t;
    if ( geom->mesh )
    {
        t.reserve( geom->mesh->num_faces() );
        for ( draco::FaceIndex f( 0 ); f < geom->mesh->num_faces(); ++f )
        {
            const auto& face = geom->mesh->face( f );
            t.push_back( { pointToVert[face[0].value()], pointToVert[face[1].value()], pointToVert[face[2].value()] } );
        }
    }

    std::vector<MeshBuilder::VertDuplication> dups;
    auto mesh = Mesh::fromTrianglesDuplicatingNonManifoldVertices( std::move( points ), t, &dups,
        { .skippedFaceCount = settings.skippedFaceCount } );
    if ( settings.duplicatedVertexCount )
        *settings.duplicatedVertexCount = int( dups.size() );
    auto copyDupAttributes = [&dups, sz = mesh.points.size()] ( auto& attributes )
    {
        if ( dups.empty() || attributes.empty() )
            return;
        attributes.resize( sz );
        for ( const auto & [src, dup] : dups )
            attributes[dup] = attributes[src];
    };
    copyDupAttributes( colors );
    copyDupAttributes( normals );
    copyDupAttributes( uvCoords );
    if ( colorAtt )
        *settings.colors = std::move( colors );
    if ( normalAtt )
        *settings.normals = std::move( normals );
    if ( uvAtt )
        *settings.uvCoords = std::move( uvCoords );

    if ( !reportProgress( settings.callback, 1.0f ) )
        return unexpectedOperationCanceled();
    return mesh;
}

MR_ADD_MESH_LOADER( IOFilter( "Google Draco (.drc)", "*.drc" ), fromDrc )

} // namespace MeshLoad

namespace MeshSave
{

Expected<void> toDrc( const Mesh& mesh, const std::filesystem::path& file, const DracoSaveOptions& options )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toDrc( mesh, out, options );
}

Expected<void> toDrc( const Mesh& mesh, std::ostream& out, const DracoSaveOptions& options )
{
    MR_TIMER;

    const auto& validVerts = mesh.topology.getValidVerts();
    const VertRenumber vertRenumber( validVerts, options.onlyValidPoints );
    std::vector<VertId> verts;
    verts.reserve( vertRenumber.sizeVerts() );
    if ( options.onlyValidPoints )
        for ( auto v : validVerts )
            verts.push_back( v );
    else
        for ( VertId v( 0 ); v < vertRenumber.sizeVerts(); ++v )
            verts.push_back( v );

    draco::Mesh dm;
    dm.set_num_points( uint32_t( verts.size() ) );
    addAttribute<Vector3f>( dm, draco::GeometryAttribute::POSITION, 3, draco::DT_FLOAT32, verts,
        [&] ( VertId v ) { return applyFloat( options.xf, mesh.points[v] ); } );
    if ( options.colors && !options.colors->empty() )
        addAttribute<Color>( dm, draco::GeometryAttribute::COLOR, 4, draco::DT_UINT8, verts,
            [&] ( VertId v ) { return getAt( *options.colors, v ); } );
    if ( options.uvMap && !options.uvMap->empty() )
        addAttribute<UVCoord>( dm, draco::GeometryAttribute::TEX_COORD, 2, draco::DT_FLOAT32, verts,
            [&] ( VertId v ) { return getAt( *options.uvMap, v ); } );

    dm.SetNumFaces( mesh.topology.numValidFaces() );
    draco::FaceIndex df( 0 );
    for ( auto f : mesh.topology.getValidFaces() )
    {
        VertId v[3];
        mesh.topology.getTriVerts( f, v );
        dm.SetFace( df++, { draco::PointIndex( vertRenumber( v[0] ) ), draco::PointIndex( vertRenumber( v[1] ) ), draco::PointIndex( vertRenumber( v[2] ) ) } );
    }

    return encode( dm, &dm, out, options );
}

Expected<void> toDrc( const Mesh& mesh, const std::filesystem::path& file, const SaveSettings& settings )
{
    return toDrc( mesh, file, DracoSaveOptions{ settings } );
}

Expected<void> toDrc( const Mesh& mesh, std::ostream& out, const SaveSettings& settings )
{
    return toDrc( mesh, out, DracoSaveOptions{ settings } );
}

MR_ADD_MESH_SAVER( IOFilter( "Google Draco (.drc)", "*.drc" ), toDrc, { .storesVertexColors = true } )

} // namespace MeshSave

namespace PointsLoad
{

Expected<PointCloud> fromDrc( const std::filesystem::path& file, const PointsLoadSettings& settings )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromDrc( in, settings ), file );
}

Expected<PointCloud> fromDrc( std::istream& in, const PointsLoadSettings& settings )
{
    MR_TIMER;
    auto geom = decode( in, settings.callback );
    if ( !geom )
        return unexpected( std::move( geom.error() ) );
    const auto& pc = *geom->cloud;

    const auto& posAtt = *pc.GetNamedAttribute( draco::GeometryAttribute::POSITION );
    const auto* colorAtt = settings.colors ? pc.GetNamedAttribute( draco::GeometryAttribute::COLOR ) : nullptr;
    const auto* normalAtt = pc.GetNamedAttribute( draco::GeometryAttribute::NORMAL );

    const auto numPoints = pc.num_points();
    PointCloud cloud;
    cloud.points.resize( numPoints );
    cloud.validPoints.resize( numPoints, true );
    if ( normalAtt )
        cloud.normals.resize( numPoints );
    if ( colorAtt )
        settings.colors->resize( numPoints );
    for ( draco::PointIndex p( 0 ); p < numPoints; ++p )
    {
        const VertId v( p.value() );
        cloud.points[v] = getVector3( posAtt, p );
        if ( normalAtt )
            cloud.normals[v] = getVector3( *normalAtt, p );
        if ( colorAtt )
            ( *settings.colors )[v] = getColor( *colorAtt, p );
    }

    if ( !reportProgress( settings.callback, 1.0f ) )
        return unexpectedOperationCanceled();
    return cloud;
}

MR_ADD_POINTS_LOADER( IOFilter( "Google Draco (.drc)", "*.drc" ), fromDrc )

} // namespace PointsLoad

namespace PointsSave
{

Expected<void> toDrc( const PointCloud& cloud, const std::filesystem::path& file, const DracoSaveOptions& options )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toDrc( cloud, out, options );
}

Expected<void> toDrc( const PointCloud& cloud, std::ostream& out, const DracoSaveOptions& options )
{
    MR_TIMER;

    std::vector<VertId> verts;
    if ( options.onlyValidPoints )
    {
        verts.reserve( cloud.validPoints.count() );
        for ( auto v : cloud.validPoints )
            verts.push_back( v );
    }
    else
    {
        verts.reserve( cloud.points.size() );
        for ( VertId v( 0 ); v < cloud.points.size(); ++v )
            verts.push_back( v );
    }
    if ( verts.empty() )
        return unexpected( "Cannot save empty point cloud in Draco format" );

    draco::PointCloud pc;
    pc.set_num_points( uint32_t( verts.size() ) );
    addAttribute<Vector3f>( pc, draco::GeometryAttribute::POSITION, 3, draco::DT_FLOAT32, verts,
        [&] ( VertId v ) { return applyFloat( options.xf, cloud.points[v] ); } );
    if ( cloud.hasNormals() )
    {
        std::optional<Matrix3d> normXf;
        if ( options.xf )
            normXf = options.xf->A.inverse().transposed();
        addAttribute<Vector3f>( pc, draco::GeometryAttribute::NORMAL, 3, draco::DT_FLOAT32, verts,
            [&] ( VertId v ) { return applyFloat( normXf ? &*normXf : nullptr, cloud.normals[v] ); } );
    }
    if ( options.colors && !options.colors->empty() )
        addAttribute<Color>( pc, draco::GeometryAttribute::COLOR, 4, draco::DT_UINT8, verts,
            [&] ( VertId v ) { return getAt( *options.colors, v ); } );

    return encode( pc, nullptr, out, options );
}

Expected<void> toDrc( const PointCloud& cloud, const std::filesystem::path& file, const SaveSettings& settings )
{
    return toDrc( cloud, file, DracoSaveOptions{ settings } );
}

Expected<void> toDrc( const PointCloud& cloud, std::ostream& out, const SaveSettings& settings )
{
    return toDrc( cloud, out, DracoSaveOptions{ settings } );
}

MR_ADD_POINTS_SAVER( IOFilter( "Google Draco (.drc)", "*.drc" ), toDrc )

} // namespace PointsSave

} // namespace MR
#endif
