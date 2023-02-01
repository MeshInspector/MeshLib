#pragma once

#include "MRMeshFwd.h"
#include "MRVector3.h"
#include "MRObject.h"
#include "MRVector4.h"
#include "MRColor.h"
#include "MRPlane3.h"
#include "MRIOFilters.h"
#include "MRProgressCallback.h"
#include <tl/expected.hpp>

namespace Json
{
class Value;
}

namespace MR
{

/// \defgroup SerializerGroup Serializer
/// \ingroup IOGroup
/// \{

/// this callback will be called before compression on serialization and after decompression on deserialization
using FolderCallback = std::function<void( const std::filesystem::path& tempFolderName )>;

class UniqueTemporaryFolder
{
public:
    /// creates new folder in temp directory
    MRMESH_API UniqueTemporaryFolder( FolderCallback onPreTempFolderDelete );
    /// removes folder with all its content
    MRMESH_API ~UniqueTemporaryFolder();

    explicit operator bool() const
    {
        return !folder_.empty();
    }
    operator const std::filesystem::path& ( ) const
    {
        return folder_;
    }
    std::filesystem::path operator /( const std::filesystem::path& child ) const
    {
        return folder_ / child;
    }

private:
    std::filesystem::path folder_;
    FolderCallback onPreTempFolderDelete_;
};

MRMESH_API extern const IOFilters SceneFileFilters;

MRMESH_API tl::expected<Json::Value, std::string> deserializeJsonValue( const std::filesystem::path& path );

/**
 * \brief saves object subtree in given scene file (zip/mru)
 * \details format specification:
 *  children are saved under folder with name of their parent object
 *  all objects parameters are saved in one JSON file in the root folder
 *  
 * if preCompress is set, it is called before compression
 * saving is controlled with Object::serializeModel_ and Object::serializeFields_
 */
MRMESH_API tl::expected<void, std::string> serializeObjectTree( const Object& object, 
    const std::filesystem::path& path, ProgressCallback progress = {}, FolderCallback preCompress = {} );
/**
 * \brief loads objects tree from given scene file (zip/mru)
 * \details format specification:
 *  children are saved under folder with name of their parent object
 *  all objects parameters are saved in one JSON file in the root folder
 *  
 * if postDecompress is set, it is called after decompression
 * loading is controlled with Object::deserializeModel_ and Object::deserializeFields_
 */
MRMESH_API tl::expected<std::shared_ptr<Object>, std::string> deserializeObjectTree( const std::filesystem::path& path,
    FolderCallback postDecompress = {}, ProgressCallback progressCb = {} );

/**
 * \brief loads objects tree from given scene folder
 * \details format specification:
 *  children are saved under folder with name of their parent object
 *  all objects parameters are saved in one JSON file in the root folder
 *  
 * loading is controlled with Object::deserializeModel_ and Object::deserializeFields_
 */
MRMESH_API tl::expected<std::shared_ptr<Object>, std::string> deserializeObjectTreeFromFolder( const std::filesystem::path& folder,
    ProgressCallback progressCb = {} );

/**
 * \brief decompresses given zip-file into given folder
 * \param password if password is given then it will be used to decipher encrypted archive
 */
MRMESH_API tl::expected<void, std::string> decompressZip( const std::filesystem::path& zipFile, const std::filesystem::path& targetFolder,
    const char * password = nullptr );
/**
 * \brief compresses given folder in given zip-file
 * \param excludeFiles files that should not be included to result zip 
 * \param password if password is given then the archive will be encrypted
 */
MRMESH_API tl::expected<void, std::string> compressZip( const std::filesystem::path& zipFile, const std::filesystem::path& sourceFolder, 
    const std::vector<std::filesystem::path>& excludeFiles = {}, const char * password = nullptr );
/// saves mesh with optional selection to mru format
MRMESH_API tl::expected<void, std::string> serializeMesh( const Mesh& mesh, const std::filesystem::path& path, const FaceBitSet* selection = nullptr );

/// saves an object into json value
MRMESH_API void serializeToJson( const Vector2i& vec, Json::Value& root );
MRMESH_API void serializeToJson( const Vector2f& vec, Json::Value& root );
MRMESH_API void serializeToJson( const Vector3i& vec, Json::Value& root );
MRMESH_API void serializeToJson( const Vector3f& vec, Json::Value& root );
MRMESH_API void serializeToJson( const Vector4f& vec, Json::Value& root );
MRMESH_API void serializeToJson( const Color& col, Json::Value& root );
MRMESH_API void serializeToJson( const Matrix3f& matrix, Json::Value& root, bool skipIdentity = true );
MRMESH_API void serializeToJson( const AffineXf3f& xf, Json::Value& root, bool skipIdentity = true );
MRMESH_API void serializeToJson( const BitSet& bitset, Json::Value& root );
MRMESH_API tl::expected<void, std::string> serializeToJson( const Mesh& mesh, Json::Value& root );
MRMESH_API void serializeToJson( const Plane3f& plane, Json::Value& root );
MRMESH_API void serializeToJson( const TriPointf& tp, Json::Value& root );
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
MRMESH_API void deserializeFromJson( const Json::Value& root, Matrix3f& matrix );
MRMESH_API void deserializeFromJson( const Json::Value& root, AffineXf3f& xf );
MRMESH_API void deserializeFromJson( const Json::Value& root, BitSet& bitset );
MRMESH_API tl::expected<Mesh, std::string> deserializeFromJson( const Json::Value& root, Vector<Color, VertId>* colors = nullptr );
MRMESH_API void deserializeFromJson( const Json::Value& root, Plane3f& plane );
MRMESH_API void deserializeFromJson( const Json::Value& root, TriPointf& tp );
/// this version takes topology to construct MeshTriPoint from its representation relative a face;
/// also beware that de-serialization will work only if faces are not renumbered (so please pack mesh before saving)
MRMESH_API void deserializeFromJson( const Json::Value& root, MeshTriPoint& mtp, const MeshTopology& topology );
MRMESH_API void deserializeFromJson( const Json::Value& root, PointOnFace& pf );

/// \}

} // namespace MR
