#include "MRMeshSave.h"
#include "MRIOFormatsRegistry.h"
#include "MRIOFilters.h"
#include "MRStringConvert.h"
#include "MRMesh.h"
#include "MRUniqueTemporaryFolder.h"
#include "MRZip.h"
#include "MRTimer.h"
#include "MRPch/MRFmt.h"

namespace MR
{

namespace MeshSave
{

static constexpr const char * sUnitNames[(int)LengthUnit::_count] =
{
    "micron",
    "millimeter",
    "centimeter",
    "meter",
    "inch",
    "foot"
};

Expected<void> toModel3mf( const Mesh & mesh, const std::filesystem::path & file, const SaveSettings & settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toModel3mf( mesh, out, settings );
}

Expected<void> toModel3mf( const Mesh & mesh, std::ostream & out, const SaveSettings & settings )
{
    MR_TIMER;

    assert( !settings.lengthUnit || *settings.lengthUnit < LengthUnit::_count );
    const char * unitName = 
        sUnitNames[int( settings.lengthUnit ? *settings.lengthUnit : LengthUnit::millimeters )];

    out <<
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<model unit=\"" << unitName << "\" xml:lang=\"en-US\" xmlns=\"http://schemas.microsoft.com/3dmanufacturing/2013/01\">\n"
        "  <resources>\n";

    if ( settings.solidColor )
    {
        out <<
            fmt::format(
            "    <m:colorgroup id=\"1\">\n"
            "      <m:color color=\"#{:02X}{:02X}{:02X}{:02X}\" />\n"
            "    </m:colorgroup>\n"
            "    <object id=\"0\" type=\"model\" pid=\"1\" pindex=\"0\">\n",
                settings.solidColor->r, settings.solidColor->g, settings.solidColor->b, settings.solidColor->a );
    }
    else
        out << "    <object id=\"0\" type=\"model\">\n";

    const VertRenumber vertRenumber( mesh.topology.getValidVerts(), settings.onlyValidPoints );
    const int numPoints = vertRenumber.sizeVerts();
    const VertId lastVertId = mesh.topology.lastValidVert();

    // write vertices
    out <<
        "      <mesh>\n"
        "        <vertices>\n";
    int numSaved = 0;
    for ( VertId i{ 0 }; i <= lastVertId; ++i )
    {
        if ( settings.onlyValidPoints && !mesh.topology.hasVert( i ) )
            continue;
        const Vector3f& p = mesh.points[i];
        out << fmt::format( "          <vertex x=\"{}\" y=\"{}\" z=\"{}\" />\n", p.x, p.y, p.z );
        ++numSaved;
        if ( settings.progress && !( numSaved & 0x3FF ) && !settings.progress( float( numSaved ) / numPoints * 0.5f ) )
            return unexpectedOperationCanceled();
    }

    out <<
        "        </vertices>\n"
        "        <triangles>\n";

    const auto fLast = mesh.topology.lastValidFace();
    const auto numSaveFaces = settings.packPrimitives ? mesh.topology.numValidFaces() : int( fLast + 1 );

    // write triangles
    int savedFaces = 0;
    for ( FaceId f{0}; f <= fLast; ++f )
    {
        int v[3] = { 0, 0, 0 };
        if ( mesh.topology.hasFace( f ) )
        {
            VertId vs[3];
            mesh.topology.getTriVerts( f, vs );
            for ( int i = 0; i < 3; ++i )
                v[i] = vertRenumber( vs[i] );
        }
        else if ( settings.packPrimitives )
            continue;
        out << fmt::format( "          <triangle v1=\"{}\" v2=\"{}\" v3=\"{}\" />\n", v[0], v[1], v[2] );
        ++savedFaces;
        if ( settings.progress && !( savedFaces & 0x3FF ) && !settings.progress( float( savedFaces ) / numSaveFaces * 0.5f + 0.5f ) )
            return unexpectedOperationCanceled();
    }

    out <<
        "        </triangles>\n"
        "      </mesh>\n"
        "    </object>\n"
        "  </resources>\n";

    AffineXf3d xf;
    if ( settings.xf )
        xf = *settings.xf;
    const auto & A = xf.A;
    const auto & b = xf.b;
    out << fmt::format(
        "  <build>\n"
        "    <item objectid=\"0\" transform=\"{} {} {} {} {} {} {} {} {} {} {} {}\" />\n"
        "  </build>\n", A.x.x, A.y.x, A.z.x,
                        A.x.y, A.y.y, A.z.y,
                        A.x.z, A.y.z, A.z.z,
                        b.x,   b.y,   b.z );

    out <<
        "</model>\n";

    if ( !out )
        return unexpected( std::string( "Error saving in 3MF model-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

Expected<void> to3mf( const Mesh & mesh, const std::filesystem::path& file, const SaveSettings & settings )
{
    MR_TIMER;
    if ( file.empty() )
        return unexpected( "Cannot save to empty path" );

    UniqueTemporaryFolder scenePath;
    if ( !scenePath )
        return unexpected( "Cannot create temporary folder" );

    {
        std::ofstream f( scenePath / "[Content_Types].xml" );
        f <<
R"(<?xml version="1.0" encoding="utf-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml" />
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml" />
  <Default Extension="texture" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodeltexture" />
  <Default Extension="xml" ContentType="application/vnd.ms-printing.printticket+xml" />
  <Default Extension="prop" ContentType="application/vnd.openxmlformats-package.core-properties+xml" />
  <Default Extension="gif" ContentType="image/gif" />
  <Default Extension="jpg" ContentType="image/jpeg" />
  <Default Extension="png" ContentType="image/png" />
</Types>
)";
        if ( !f )
            return unexpected( "Cannot write file inside temporary folder" );
    }

    {
        const auto dirRels = scenePath / "_rels";
        std::error_code ec;
        if ( !create_directory( dirRels, ec ) )
            return unexpected( "Cannot create subdirectory inside temporary folder" );

        std::ofstream f( dirRels / ".rels" );
        f <<
R"(<?xml version="1.0" encoding="utf-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" Target="/3D/3dmodel.model" Id="rel0" />
</Relationships>
)";
        if ( !f )
            return unexpected( "Cannot write file inside temporary folder" );
    }

    {
        const auto dir3D = scenePath / "3D";
        std::error_code ec;
        if ( !create_directory( dir3D, ec ) )
            return unexpected( "Cannot create subdirectory inside temporary folder" );

        auto settings1 = settings;
        settings1.progress = subprogress( settings.progress, 0.0f, 0.9f );
        if ( auto maybeDone = toModel3mf( mesh, dir3D / "3dmodel.model", settings1 ); !maybeDone )
            return unexpected( std::move( maybeDone.error() ) );
    }

    return compressZip( file, scenePath, {}, nullptr, subprogress( settings.progress, 0.9f, 1.0f ) );
}

MR_ADD_MESH_SAVER( IOFilter( "3D Manufacturing model (.model)", "*.model" ), toModel3mf, {} )

MR_ON_INIT { using namespace MR::MeshSave; setMeshSaver( IOFilter( "3D Manufacturing format (.3mf)", "*.3mf" ), { to3mf, nullptr, {} } ); };

} //namespace MeshSave

} //namespace MR
