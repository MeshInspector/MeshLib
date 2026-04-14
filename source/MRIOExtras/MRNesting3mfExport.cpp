#include "MRNesting3mfExport.h"
#ifndef MRIOEXTRAS_NO_XML
#include "MRMesh/MRUniqueTemporaryFolder.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRId.h"
#include "MRMesh/MRProgressCallback.h"
#include "MRMesh/MRParallelFor.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRExtractIsolines.h"
#include "MRMesh/MRPolylineDecimate.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRZip.h"
#include "MRMesh/MRImage.h"
#include "MRMesh/MRImageSave.h"
#include "MRPch/MRFmt.h"
#include "MRPch/MRTBB.h"
#include "MRMesh/MR2to3.h"
#include <tinyxml2.h>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/uuid_generators.hpp>

#include <filesystem>

namespace
{
std::string generateUUID()
{
    boost::uuids::random_generator generator;
    boost::uuids::uuid u = generator();
    return boost::uuids::to_string( u );
}
}

namespace MR::Nesting
{

using Slices2D = std::vector<std::pair<Contours2f, float>>;

static Slices2D prepareSlices( const MeshXf& mesh, const Box3f& worldBox, float step, bool decimate )
{
    if ( !mesh.mesh )
        return {};
    auto invXf = mesh.xf.inverse();
    auto minDist = worldBox.min.z;
    auto maxDist = worldBox.max.z - minDist;
    int numSlices = 1; // one empty slice at the end
    for ( auto dist = step * 0.5f; dist <= maxDist; dist += step )
        ++numSlices;

    Slices2D res;
    res.resize( numSlices );

    ParallelFor( res, [&] ( size_t i )
    {
        auto& [slices, ztop] = res[i];
        ztop = ( i + 1 ) * step;

        auto plane = transformed( -Plane3f( Vector3f::plusZ(), minDist + ztop - step * 0.5f ), invXf ).normalized();
        auto sections = extractPlaneSections( *mesh.mesh, plane );
        if ( sections.empty() )
            return;

        if ( decimate )
        {
            Contours3f conts;
            Polyline3 polyline;
            for ( const auto& section : sections )
                polyline.addFromSurfacePath( *mesh.mesh, section );
            DecimatePolylineSettings3 pds;
            pds.maxError = 0.5f * step;
            decimatePolyline( polyline, pds );
            conts = polyline.contours();
            slices.resize( conts.size() );

            for ( int si = 0; si < conts.size(); ++si )
            {
                auto& sliceI = slices[si];
                const auto& contI = conts[si];
                sliceI.resize( contI.size() );
                for ( int sj = 0; sj < contI.size(); ++sj )
                    sliceI[sj] = to2dim( mesh.xf( contI[sj] ) - worldBox.min );
            }
        }
        else
        {
            for ( int si = 0; si < sections.size(); ++si )
            {
                auto& sliceI = slices[si];
                const auto& sectionI = sections[si];
                sliceI.resize( sectionI.size() );
                for ( int sj = 0; sj < sectionI.size(); ++sj )
                    sliceI[sj] = to2dim( mesh.xf( mesh.mesh->edgePoint( sectionI[sj] ) ) - worldBox.min );
            }
        }
    } );
    return res;
}

void save2dModelFile( const std::filesystem::path& path, const Slices2D& slices )
{
    if ( slices.empty() )
        return;

    tinyxml2::XMLDocument slicesDoc;
    auto decl = slicesDoc.NewDeclaration();
    slicesDoc.LinkEndChild( decl );
    auto model = slicesDoc.NewElement( "model" );
    model->SetAttribute( "xmlns", "http://schemas.microsoft.com/3dmanufacturing/core/2015/02" );
    model->SetAttribute( "xmlns:s", "http://schemas.microsoft.com/3dmanufacturing/slice/2015/07" );
    model->SetAttribute( "xmlns:p", "http://schemas.microsoft.com/3dmanufacturing/production/2015/06" );
    model->SetAttribute( "unit", "millimeter" );
    model->SetAttribute( "xml:lang", "en-US" );
    auto resources = model->InsertNewChildElement( "resources" );
    auto slicestack = resources->InsertNewChildElement( "s:slicestack" );
    slicestack->SetAttribute( "id", 1 );
    auto tempStr = fmt::format( "{:.3f}", 0.0f );
    slicestack->SetAttribute( "zbottom", tempStr.c_str() );

    for ( int i = 0; i < slices.size(); ++i )
    {
        const auto& [zSlices, ztop] = slices[i];
        auto slice = slicestack->InsertNewChildElement( "s:slice" );
        tempStr = fmt::format( "{:.3f}", ztop );
        slice->SetAttribute( "ztop", tempStr.c_str() );
        if ( !zSlices.empty() )
        {
            auto vertices = slice->InsertNewChildElement( "s:vertices" );

            for ( const auto& zSlice : zSlices )
            {
                for ( int si = 0; si + 1 < zSlice.size(); ++si )
                {
                    auto coord = zSlice[si];

                    auto vertex = vertices->InsertNewChildElement( "s:vertex" );
                    tempStr = fmt::format( "{:.3f}", coord.x );
                    vertex->SetAttribute( "x", tempStr.c_str() );
                    tempStr = fmt::format( "{:.3f}", coord.y );
                    vertex->SetAttribute( "y", tempStr.c_str() );
                    vertices->LinkEndChild( vertex );
                }
            }
            slice->LinkEndChild( vertices );
            int numV = -1;
            for ( const auto& zSlice : zSlices )
            {
                auto polygon = slice->InsertNewChildElement( "s:polygon" );
                auto startV = ++numV;
                polygon->SetAttribute( "startv", startV );
                for ( int si = 1; si < zSlice.size(); ++si )
                {
                    auto segment = polygon->InsertNewChildElement( "s:segment" );
                    if ( si + 1 < zSlice.size() )
                        segment->SetAttribute( "v2", ++numV );
                    else
                        segment->SetAttribute( "v2", startV );
                    polygon->LinkEndChild( segment );
                }
                slice->LinkEndChild( polygon );
            }
        }
        slicestack->LinkEndChild( slice );
    }

    resources->LinkEndChild( slicestack );
    model->LinkEndChild( resources );
    model->LinkEndChild( model->InsertNewChildElement( "build" ) );
    slicesDoc.LinkEndChild( model );
    auto slPath = utf8string( path );
    slicesDoc.SaveFile( slPath.c_str() );
}

Expected<void> exportNesting3mf( const std::filesystem::path& path, const Nesting3mfParams& params )
{
    if ( params.zStep <= 0 )
    {
        assert( false );
        return unexpected( "Negative slice step" );
    }
    UniqueTemporaryFolder saveDir;
    std::filesystem::path dir = saveDir;

    std::error_code ec;
    if ( !std::filesystem::is_directory( dir, ec ) && !std::filesystem::create_directories( dir, ec ) )
        return unexpected( systemToUtf8( ec.message() ) );
    auto dir2d = dir / "2D";
    auto dir3d = dir / "3D";
    auto dirRels = dir / "_rels";
    auto dir3dRels = dir3d / "_rels";
    auto dirMeta = dir / "Metadata";
    auto dirThumb = dir / "Thumbnails";
    if ( !std::filesystem::is_directory( dir2d, ec ) && !std::filesystem::create_directories( dir2d, ec ) )
        return unexpected( systemToUtf8( ec.message() ) );
    if ( !std::filesystem::is_directory( dir3d, ec ) && !std::filesystem::create_directories( dir3d, ec ) )
        return unexpected( systemToUtf8( ec.message() ) );
    if ( !std::filesystem::is_directory( dirRels, ec ) && !std::filesystem::create_directories( dirRels, ec ) )
        return unexpected( systemToUtf8( ec.message() ) );
    if ( !std::filesystem::is_directory( dir3dRels, ec ) && !std::filesystem::create_directories( dir3dRels, ec ) )
        return unexpected( systemToUtf8( ec.message() ) );
    if ( params.image && !std::filesystem::is_directory( dirMeta, ec ) && !std::filesystem::create_directories( dirMeta, ec ) )
        return unexpected( systemToUtf8( ec.message() ) );
    if ( params.meshImages && !std::filesystem::is_directory( dirThumb, ec ) && !std::filesystem::create_directories( dirThumb, ec ) )
        return unexpected( systemToUtf8( ec.message() ) );

    // [Content_Types].xml
    {
        tinyxml2::XMLDocument contentType;
        auto decl = contentType.NewDeclaration("xml version=\"1.0\"");
        contentType.LinkEndChild( decl );

        auto* type = contentType.NewElement( "Types" );
        type->SetAttribute( "xmlns", "http://schemas.openxmlformats.org/package/2006/content-types" );
        auto el = type->InsertNewChildElement( "Default" );
        el->SetAttribute( "Extension", "rels" );
        el->SetAttribute( "ContentType", "application/vnd.openxmlformats-package.relationships+xml" );
        type->LinkEndChild( el );
        el = type->InsertNewChildElement( "Default" );
        el->SetAttribute( "Extension", "model" );
        el->SetAttribute( "ContentType", "application/vnd.ms-package.3dmanufacturing-3dmodel+xml" );
        type->LinkEndChild( el );
        if ( params.image )
        {
            el = type->InsertNewChildElement( "Default" );
            el->SetAttribute( "Extension", "png" );
            el->SetAttribute( "ContentType", "image/png" );
            type->LinkEndChild( el );
        }
        contentType.LinkEndChild( type );
        auto ctPath = utf8string( dir / "[Content_Types].xml" );
        contentType.SaveFile( ctPath.c_str() );
    }

    if ( !reportProgress( params.cb, 0.05f ) )
        return unexpectedOperationCanceled();

    Vector<Box3f, ObjId> worldBoxes( params.meshes.size() );
    auto keepGoing = ParallelFor( params.meshes, [&] ( ObjId oid )
    {
        if ( !params.meshes[oid].mesh )
            return;
        worldBoxes[oid] = params.meshes[oid].mesh->computeBoundingBox( &params.meshes[oid].xf );
    }, subprogress( params.cb, 0.05f, 0.1f ) );

    if ( !keepGoing )
        return unexpectedOperationCanceled();

    std::vector<std::string> uuids( params.meshes.size() );
    for ( int i = 0; i < uuids.size(); ++i )
        uuids[i] = generateUUID();

    // Thumbnails
    if ( params.meshImages )
    {
        auto sb = subprogress( params.cb, 0.1f, 0.15f );
        if ( params.meshImages->size() != params.meshes.size() )
            return unexpected( "Invalid number of one by one screenshots" );
        for ( int i = 0; i < params.meshes.size(); ++i )
        {
            auto saveRes = ImageSave::toAnySupportedFormat( ( *params.meshImages )[i], dirThumb / ( uuids[i] + ".png" ) );
            if ( !saveRes.has_value() )
                return unexpected( "Cannot save screenshot: " + saveRes.error() );
            if ( !reportProgress( sb, float( i + 1 ) / float( params.meshes.size() ) ) )
                return unexpectedOperationCanceled();
        }
    }

    // 2D
    {
        keepGoing = ParallelFor( params.meshes, [&] ( ObjId i )
        {
            auto slices = prepareSlices( params.meshes[i], worldBoxes[i], params.zStep, params.decimateSlices );
            auto slPath = utf8string( dir / "2D" / ( uuids[i] + ".model" ) );
            save2dModelFile( slPath, slices );
        }, subprogress( params.cb, 0.15f, 0.4f ) );
        if ( !keepGoing )
            return unexpectedOperationCanceled();
    }

    std::string tempStr;
    // 3D
    {
        tinyxml2::XMLDocument buildDoc;
        auto decl = buildDoc.NewDeclaration();
        buildDoc.LinkEndChild( decl );
        auto model = buildDoc.NewElement( "model" );
        model->SetAttribute( "xmlns", "http://schemas.microsoft.com/3dmanufacturing/core/2015/02" );
        model->SetAttribute( "xmlns:s", "http://schemas.microsoft.com/3dmanufacturing/slice/2015/07" );
        model->SetAttribute( "xmlns:p", "http://schemas.microsoft.com/3dmanufacturing/production/2015/06" );
        model->SetAttribute( "unit", "millimeter" );
        model->SetAttribute( "xml:lang", "en-US" );
        auto resources = model->InsertNewChildElement( "resources" );
        for ( ObjId i( 0 ); i < params.meshes.size(); ++i )
        {
            auto slicestack = resources->InsertNewChildElement( "s:slicestack" );
            slicestack->SetAttribute( "id", int( i + 1 ) );
            tempStr = fmt::format( "{:.3f}",  0.0f );
            slicestack->SetAttribute( "zbottom", tempStr.c_str() );
            auto sliceref = slicestack->InsertNewChildElement( "s:sliceref" );
            sliceref->SetAttribute( "slicestackid", 1 );
            auto slicepath = "/2D/" + uuids[i] + ".model";
            sliceref->SetAttribute( "slicepath", slicepath.c_str() );
            slicestack->LinkEndChild( sliceref );
            resources->LinkEndChild( slicestack );
        }

        for ( ObjId i( 0 ); i < params.meshes.size(); ++i )
        {
            auto object = resources->InsertNewChildElement( "object" );
            object->SetAttribute( "id", int( i + 1 + params.meshes.size() ) );
            auto name = std::to_string( int( i + 1 ) ) + "_printable.model";
            if ( params.meshNames )
                name = ( *params.meshNames )[int( i )];
            object->SetAttribute( "name", name.c_str() );
            auto thumbnail = "/Thumbnails/" + uuids[i] + ".png";
            object->SetAttribute( "thumbnail", thumbnail.c_str() );
            object->SetAttribute( "s:slicestackid", int( i + 1 ) );
            object->SetAttribute( "p:UUID", uuids[i].c_str() );
            auto mesh = object->InsertNewChildElement( "mesh" );
            auto vertices = mesh->InsertNewChildElement( "vertices" );
            constexpr uint8_t cOrder[8] = { 0,2,3,1,4,6,7,5 };
            for ( auto ord : cOrder )
            {
                auto pos = worldBoxes[i].corner( Vector3b( bool( ord & 1 ), bool( ord & 2 ), bool( ord & 4 ) ) ) - worldBoxes[i].min;
                auto vert = vertices->InsertNewChildElement( "vertex" );
                tempStr = fmt::format( "{:.3f}",  pos.x );
                vert->SetAttribute( "x", tempStr.c_str() );
                tempStr = fmt::format( "{:.3f}",  pos.y );
                vert->SetAttribute( "y", tempStr.c_str() );
                tempStr = fmt::format( "{:.3f}",  pos.z );
                vert->SetAttribute( "z", tempStr.c_str() );
                vertices->LinkEndChild( vert );
            }
            mesh->LinkEndChild( vertices );
            auto triangles = mesh->InsertNewChildElement( "triangles" );
            constexpr ThreeVertIds cTris[12] = {
                {VertId( 0 ),VertId( 1 ),VertId( 2 )},
                {VertId( 0 ),VertId( 2 ),VertId( 3 )},
                {VertId( 4 ),VertId( 7 ),VertId( 6 )},
                {VertId( 4 ),VertId( 6 ),VertId( 5 )},
                {VertId( 0 ),VertId( 4 ),VertId( 5 )},
                {VertId( 0 ),VertId( 5 ),VertId( 1 )},
                {VertId( 1 ),VertId( 5 ),VertId( 6 )},
                {VertId( 1 ),VertId( 6 ),VertId( 2 )},
                {VertId( 2 ),VertId( 6 ),VertId( 7 )},
                {VertId( 2 ),VertId( 7 ),VertId( 3 )},
                {VertId( 3 ),VertId( 7 ),VertId( 4 )},
                {VertId( 3 ),VertId( 4 ),VertId( 0 )}
            };
            for ( const auto& triV : cTris )
            {
                auto tri = triangles->InsertNewChildElement( "triangle" );
                tri->SetAttribute( "v1", int( triV[0] ) );
                tri->SetAttribute( "v2", int( triV[1] ) );
                tri->SetAttribute( "v3", int( triV[2] ) );
                triangles->LinkEndChild( tri );
            }
            mesh->LinkEndChild( triangles );
            object->LinkEndChild( mesh );
            resources->LinkEndChild( object );
        }
        model->LinkEndChild( resources );
        auto build = model->InsertNewChildElement( "build" );
        auto tempUUID = generateUUID();
        build->SetAttribute( "p:UUID", tempUUID.c_str() );
        for ( ObjId i( 0 ); i < params.meshes.size(); ++i )
        {
            auto item = build->InsertNewChildElement( "item" );
            item->SetAttribute( "objectid", int( i + 1 + params.meshes.size() ) );
            std::string transformStr = "1.0000 0.0000 0.0000 0.0000 1.0000 0.0000 0.0000 0.0000 1.0000 ";
            transformStr += 
                fmt::format( "{:.4f}",  worldBoxes[i].min.x ) + " " + 
                fmt::format( "{:.4f}",  worldBoxes[i].min.y ) + " " + 
                fmt::format( "{:.4f}",  worldBoxes[i].min.z );
            item->SetAttribute( "transform", transformStr.c_str() );
            tempUUID = generateUUID();
            item->SetAttribute( "p:UUID", tempUUID.c_str() );
            build->LinkEndChild( item );
        }
        model->LinkEndChild( build );
        buildDoc.LinkEndChild( model );
        auto bdPath = utf8string( dir / "3D" / "3dmodel.model" );
        buildDoc.SaveFile( bdPath.c_str() );
    }

    if ( !reportProgress( params.cb, 0.4f ) )
        return unexpectedOperationCanceled();

    // image
    if ( params.image )
    {
        auto imageSaveRes = ImageSave::toAnySupportedFormat( *params.image, dirMeta / "thumbnail.png" );
        if ( !imageSaveRes.has_value() )
            return unexpected( std::move( imageSaveRes.error() ) );
    }

    if ( !reportProgress( params.cb, 0.45f ) )
        return unexpectedOperationCanceled();
    // rels
    {
        tinyxml2::XMLDocument rootRels;
        auto decl = rootRels.NewDeclaration( "xml version=\"1.0\"" );
        rootRels.LinkEndChild( decl );
        auto rels = rootRels.NewElement( "Relationships" );
        rels->SetAttribute( "xmlns", "http://schemas.openxmlformats.org/package/2006/relationships" );
        auto rel = rels->InsertNewChildElement( "Relationship" );
        rel->SetAttribute( "Target", "/3D/3dmodel.model" );
        rel->SetAttribute( "Id", "rel0" );
        rel->SetAttribute( "Type", "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" );
        rels->LinkEndChild( rel );
        if ( params.image )
        {
            rel = rels->InsertNewChildElement( "Relationship" );
            rel->SetAttribute( "Target", "/Metadata/thumbnail.png" );
            rel->SetAttribute( "Id", "rel1" );
            rel->SetAttribute( "Type", "http://schemas.openxmlformats.org/package/2006/relationships/metadata/thumbnail" );
            rels->LinkEndChild( rel );
        }
        rootRels.LinkEndChild( rels );
        auto rootRelsPath = utf8string( dirRels / ".rels" );
        rootRels.SaveFile( rootRelsPath.c_str() );

        tinyxml2::XMLDocument rels3d;
        decl = rels3d.NewDeclaration( "xml version=\"1.0\"" );
        rels3d.LinkEndChild( decl );
        rels = rels3d.NewElement( "Relationships" );
        rels->SetAttribute( "xmlns", "http://schemas.openxmlformats.org/package/2006/relationships" );
        for ( int i = 0; i < params.meshes.size(); ++i )
        {
            rel = rels->InsertNewChildElement( "Relationship" );
            std::string slicestackName = "/2D/" + uuids[i] + ".model";
            rel->SetAttribute( "Target", slicestackName.c_str() );
            std::string relId = "rel" + std::to_string( 2 * i + 1 );
            rel->SetAttribute( "Id", relId.c_str() );
            rel->SetAttribute( "Type", "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" );
            rels->LinkEndChild( rel );

            rel = rels->InsertNewChildElement( "Relationship" );
            std::string thumbnailsName = "/Thumbnails/" + uuids[i] + ".png";
            rel->SetAttribute( "Target", thumbnailsName.c_str() );
            relId = "rel" + std::to_string( 2 * i + 2 );
            rel->SetAttribute( "Id", relId.c_str() );
            rel->SetAttribute( "Type", "http://schemas.openxmlformats.org/package/2006/relationships/metadata/thumbnail" );
            rels->LinkEndChild( rel );
        }
        rels3d.LinkEndChild( rels );
        auto rels3dPath = utf8string( dir3dRels / "3dmodel.model.rels" );
        rels3d.SaveFile( rels3dPath.c_str() );
    }

    if ( !reportProgress( params.cb, 0.5f ) )
        return unexpectedOperationCanceled();

    return compressZip( path, dir, { .compressionLevel = 2, .cb = subprogress( params.cb, 0.5f, 1.0f ) } );
}

}
#endif
