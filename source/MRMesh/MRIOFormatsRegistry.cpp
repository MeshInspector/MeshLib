#include "MRIOFormatsRegistry.h"
#include "MRImage.h"

namespace MR
{

const IOFilters AllFilter =
{
    { "All (.*)", "*.*" },
};

namespace MeshLoad
{

MR_FORMAT_REGISTRY_IMPL( MeshLoader )

} // namespace MeshLoad

namespace MeshSave
{

MR_FORMAT_REGISTRY_IMPL( MeshSaver )

} // namespace MeshSave

namespace LinesLoad
{

MR_FORMAT_REGISTRY_IMPL( LinesLoader )

} // namespace LinesLoad

namespace LinesSave
{

MR_FORMAT_REGISTRY_IMPL( LinesSaver )

} // namespace LinesSave

namespace PointsLoad
{

MR_FORMAT_REGISTRY_IMPL( PointsLoader )

} // namespace PointsLoad

namespace PointsSave
{

MR_FORMAT_REGISTRY_IMPL( PointsSaver )

} // namespace PointsSave

namespace ImageLoad
{

MR_FORMAT_REGISTRY_IMPL( ImageLoader )

} // namespace ImageLoad

namespace ImageSave
{

MR_FORMAT_REGISTRY_IMPL( ImageSaver )

} // namespace ImageSave

namespace ObjectLoad
{

MR_FORMAT_REGISTRY_IMPL( ObjectLoader )

} // namespace ObjectLoad

namespace ObjectSave
{

MR_FORMAT_REGISTRY_IMPL( ObjectSaver )

} // namespace ObjectSave

namespace AsyncObjectLoad
{

MR_FORMAT_REGISTRY_IMPL( ObjectLoader )

} // namespace AsyncObjectLoad

namespace SceneLoad
{

MR_FORMAT_REGISTRY_IMPL( SceneLoader )

} // namespace SceneLoad

namespace SceneSave
{

MR_FORMAT_REGISTRY_IMPL( SceneSaver )

} // namespace SceneSave

} // namespace MR
