#include "MRMeshSave.h"
#include "MRIOFormatsRegistry.h"
#include "MRIOFilters.h"
#include "MRStringConvert.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRPch/MRFmt.h"

namespace MR
{

namespace MeshSave
{

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

    out <<
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<model unit=\"millimeter\" xml:lang=\"en-US\" xmlns=\"http://schemas.microsoft.com/3dmanufacturing/2013/01\">\n"
        "  <resources>\n"
        "    <object id=\"0\" type=\"model\">\n"
        "      <mesh>\n"
        "        <vertices>\n";

    const VertRenumber vertRenumber( mesh.topology.getValidVerts(), settings.onlyValidPoints );
    const int numPoints = vertRenumber.sizeVerts();
    const VertId lastVertId = mesh.topology.lastValidVert();

    // write vertices
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
        "  </build>\n", A.x.x, A.x.y, A.x.z, A.y.x, A.y.y, A.y.z, A.z.x, A.z.y, A.z.z, b.x, b.y, b.z );

    out <<
        "</model>\n";

    if ( !out )
        return unexpected( std::string( "Error saving in 3MF model-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

MR_ADD_MESH_SAVER( IOFilter( "3D Manufacturing model (.model)", "*.model" ), toModel3mf, {} )

} //namespace MeshSave

} //namespace MR
