#pragma once

#include "config.h"
#include "exports.h"
#include "MRStep.h"

namespace MR
{

/// default settings of MRIOExtras formats, used when loading or saving via the format registry; thread-safe
class ExtraFormatSettings
{
public:
    /// resets all settings to default values
    MRIOEXTRAS_API static void reset();

#ifndef MRIOEXTRAS_NO_STEP
    MRIOEXTRAS_API static MeshLoad::StepLoadSettings getStepLoadSettings();
    MRIOEXTRAS_API static void setStepLoadSettings( const MeshLoad::StepLoadSettings& settings );
#endif
};

} // namespace MR
