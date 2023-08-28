#include "MRPointsLoad.h"
#include "MRPointCloud.h"
#include "MRStringConvert.h"
#include "MRTimer.h"
#include <e57/time_conversion/time_conversion.h>
#include <e57/E57Foundation.h>
#include <e57/E57Simple.h>

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
