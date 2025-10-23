#pragma once

#include "MREigenCore.h"

#pragma warning(push)
#pragma warning(disable: 4127) // conditional expression is constant

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option" // for next one
#pragma clang diagnostic ignored "-Wunused-but-set-variable" // for newer clang
#endif

#include <Eigen/SparseCore>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#pragma warning(pop)
