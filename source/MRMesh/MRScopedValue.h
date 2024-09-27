#pragma once

#include "MRFinally.h"

/// change a variable's value until the current scope's end
#define MR_SCOPED_VALUE( var, value ) \
const auto MR_CONCAT( _prev_value_, __LINE__ ) = ( var ); ( var ) = ( value ); MR_FINALLY { ( var ) = MR_CONCAT( _prev_value_, __LINE__ ); };
