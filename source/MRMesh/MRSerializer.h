#pragma once

#include "MRMeshFwd.h"
#include "MRVector2.h"
#include "MRVector3.h"
#include "MRVector4.h"
#include "MRMatrix2.h"
#include "MRColor.h"
#include "MRPlane3.h"
#include "MRIOFilters.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"
#include <filesystem>

namespace Json
{
class Value;
}

namespace MR
{

/// \defgroup SerializerGroup Serializer
/// \ingroup IOGroup
/// \{

MRMESH_API Expected<std::string> serializeJsonValue( const Json::Value& root );
MRMESH_API Expected<void> serializeJsonValue( const Json::Value& root, std::ostream& out ); // important on Windows: in stream must be open in binary mode
MRMESH_API Expected<void> serializeJsonValue( const Json::Value& root, const std::filesystem::path& path );

MRMESH_API Expected<Json::Value> deserializeJsonValue( std::istream& in ); // important on Windows: in stream must be open in binary mode
MRMESH_API Expected<Json::Value> deserializeJsonValue( const char* data, size_t size );
MRMESH_API Expected<Json::Value> deserializeJsonValue( const std::string& str );
MRMESH_API Expected<Json::Value> deserializeJsonValue( const std::filesystem::path& path );

/// saves mesh with optional selection to mru format;
/// this is very convenient for saving intermediate states during algorithm debugging;
/// ".mrmesh" save mesh format is not space efficient, but guaranties no changes in the topology after loading
MRMESH_API Expected<void> serializeMesh( const Mesh& mesh, const std::filesystem::path& path, const FaceBitSet* selection = nullptr,
    const char * serializeFormat = ".mrmesh" );

/// saves an object into json value
MRMESH_API void serializeToJson( const Vector2i& vec, Json::Value& root );
MRMESH_API void serializeToJson( const Vector2f& vec, Json::Value& root );
MRMESH_API void serializeToJson( const Vector3i& vec, Json::Value& root );
MRMESH_API void serializeToJson( const Vector3f& vec, Json::Value& root );
MRMESH_API void serializeToJson( const Vector4f& vec, Json::Value& root );
MRMESH_API void serializeToJson( const Box3i& box, Json::Value& root );
MRMESH_API void serializeToJson( const Box3f& box, Json::Value& root );
MRMESH_API void serializeToJson( const Color& col, Json::Value& root );
MRMESH_API void serializeToJson( const Matrix2f& matrix, Json::Value& root, bool skipIdentity = true );
MRMESH_API void serializeToJson( const Matrix3f& matrix, Json::Value& root, bool skipIdentity = true );
MRMESH_API void serializeToJson( const AffineXf2f& xf, Json::Value& root, bool skipIdentity = true );
MRMESH_API void serializeToJson( const AffineXf3f& xf, Json::Value& root, bool skipIdentity = true );
MRMESH_API void serializeToJson( const BitSet& bitset, Json::Value& root );
MRMESH_API Expected<void> serializeToJson( const Mesh& mesh, Json::Value& root );
MRMESH_API void serializeToJson( const Plane3f& plane, Json::Value& root );
MRMESH_API void serializeToJson( const TriPointf& tp, Json::Value& root );
MRMESH_API void serializeToJson( const MeshTexture& texture, Json::Value& root );
MRMESH_API void serializeToJson( const std::vector<TextureId>& texturePerFace, Json::Value& root );
MRMESH_API void serializeToJson( const std::vector<UVCoord>& uvCoords, Json::Value& root );
MRMESH_API void serializeToJson( const std::vector<Color>& colors, Json::Value& root );
/// this version takes topology to convert MeshTriPoint in its representation relative a face;
/// also beware that de-serialization will work only if faces are not renumbered (so please pack mesh before saving)
MRMESH_API void serializeToJson( const MeshTriPoint& mtp, const MeshTopology& topology, Json::Value& root );
MRMESH_API void serializeToJson( const PointOnFace& pf, Json::Value& root );

/// serialize given edges into json first converting them into pairs of vertices,
/// which is safer when edge ids change after saving/loading, but vertex ids are not
MRMESH_API void serializeViaVerticesToJson( const UndirectedEdgeBitSet& edges, const MeshTopology & topology, Json::Value& root );
MRMESH_API void deserializeViaVerticesFromJson( const Json::Value& root, UndirectedEdgeBitSet& edges, const MeshTopology & topology );

/// loads an object from json value
MRMESH_API void deserializeFromJson( const Json::Value& root, Vector2i& vec );
MRMESH_API void deserializeFromJson( const Json::Value& root, Vector2f& vec );
MRMESH_API void deserializeFromJson( const Json::Value& root, Vector3i& vec );
MRMESH_API void deserializeFromJson( const Json::Value& root, Vector3f& vec );
MRMESH_API void deserializeFromJson( const Json::Value& root, Vector4f& vec );
MRMESH_API void deserializeFromJson( const Json::Value& root, Color& col );
MRMESH_API void deserializeFromJson( const Json::Value& root, Matrix2f& matrix );
MRMESH_API void deserializeFromJson( const Json::Value& root, Matrix3f& matrix );
MRMESH_API void deserializeFromJson( const Json::Value& root, AffineXf2f& xf );
MRMESH_API void deserializeFromJson( const Json::Value& root, AffineXf3f& xf );
MRMESH_API void deserializeFromJson( const Json::Value& root, BitSet& bitset );
MRMESH_API Expected<Mesh> deserializeFromJson( const Json::Value& root, VertColors* colors = nullptr );
MRMESH_API void deserializeFromJson( const Json::Value& root, Plane3f& plane );
MRMESH_API void deserializeFromJson( const Json::Value& root, TriPointf& tp );
MRMESH_API void deserializeFromJson( const Json::Value& root, MeshTexture& texture );
MRMESH_API void deserializeFromJson( const Json::Value& root, std::vector<TextureId>& texturePerFace );
MRMESH_API void deserializeFromJson( const Json::Value& root, std::vector<UVCoord>& uvCoords );
MRMESH_API void deserializeFromJson( const Json::Value& root, std::vector<Color>& colors );
/// this version takes topology to construct MeshTriPoint from its representation relative a face;
/// also beware that de-serialization will work only if faces are not renumbered (so please pack mesh before saving)
MRMESH_API void deserializeFromJson( const Json::Value& root, MeshTriPoint& mtp, const MeshTopology& topology );
MRMESH_API void deserializeFromJson( const Json::Value& root, PointOnFace& pf );

/// \}

} // namespace MR
