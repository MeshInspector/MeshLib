#include "MRPointsLoad.h"
#include "MRPointCloud.h"
#include "MRStringConvert.h"
#include "MRTimer.h"

#pragma warning(push)
#pragma warning(disable: 4251) // class needs to have dll-interface to be used by clients of another class
#include <E57Format/E57SimpleReader.h>
#pragma warning(pop)

#pragma warning(disable: 4996) // allow using deprecated functions

namespace MR
{

namespace PointsLoad
{

Expected<PointCloud, std::string> fromE57( std::istream& /*in*/, ProgressCallback /*callback*/ )
{
    assert( false );
    PointCloud res;
    return res;
}

Expected<PointCloud, std::string> fromE57( const std::filesystem::path& file, ProgressCallback /*callback*/ )
{
    MR_TIMER

    PointCloud res;
    e57::Reader eReader( utf8string( file ) );

    int scanIndex = 0;

    e57::Data3D scanHeader;
    eReader.ReadData3D( scanIndex, scanHeader );

    return res;
}

} //namespace PointsLoad

} //namespace MR
