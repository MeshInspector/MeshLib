#include "MRSignDetectionMode.h"
#include <cassert>

namespace MR
{

const char * asString( SignDetectionMode m )
{
    switch ( m )
    {
    case SignDetectionMode::Unsigned:
        return "Unsigned";
    case SignDetectionMode::OpenVDB:
        return "OpenVDB";
    case SignDetectionMode::ProjectionNormal:
        return "ProjectionNormal";
    case SignDetectionMode::OddCrossings:
        return "OddCrossings";
    case SignDetectionMode::HoleWindingRule:
        return "HoleWindingRule";
    default:
        assert( false );
        return "";
    }
}

} //namespace MR
