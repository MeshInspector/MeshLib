#pragma once

#include "MRFinally.h"

/// change a variable's value until the current scope's end
#define MR_SCOPED_VALUE( var, ... ) \
auto MR_CONCAT( _prev_value_, __LINE__ ) = std::move( var ); ( var ) = ( __VA_ARGS__ ); MR_FINALLY { ( var ) = std::move( MR_CONCAT( _prev_value_, __LINE__ ) ); }
