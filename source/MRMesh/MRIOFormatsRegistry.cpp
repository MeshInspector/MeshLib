#include "MRIOFormatsRegistry.h"

namespace MR
{

const IOFilters AllFilter =
{
    { "All (*.*)", "*.*" },
};

namespace MeshLoad
{

MR_FORMAT_REGISTRY_IMPL( MeshLoader )

} // namespace MeshLoad

namespace ObjectLoad
{

MR_FORMAT_REGISTRY_IMPL( ObjectLoader )

} // namespace ObjectLoad

namespace AsyncObjectLoad
{

MR_FORMAT_REGISTRY_IMPL( ObjectLoader )

} // namespace AsyncObjectLoad

} // namespace MR
