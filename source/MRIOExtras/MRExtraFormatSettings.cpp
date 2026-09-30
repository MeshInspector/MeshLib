#include "MRExtraFormatSettings.h"

#include <mutex>

namespace MR
{

namespace
{

struct Settings
{
#ifndef MRIOEXTRAS_NO_STEP
    MeshLoad::StepLoadSettings step;
#endif
};

std::mutex sMutex;
Settings sSettings;

} // namespace

void ExtraFormatSettings::reset()
{
    std::unique_lock lock( sMutex );
    sSettings = {};
}

#ifndef MRIOEXTRAS_NO_STEP
MeshLoad::StepLoadSettings ExtraFormatSettings::getStepLoadSettings()
{
    std::unique_lock lock( sMutex );
    return sSettings.step;
}

void ExtraFormatSettings::setStepLoadSettings( const MeshLoad::StepLoadSettings& settings )
{
    std::unique_lock lock( sMutex );
    sSettings.step = settings;
}
#endif

} // namespace MR
