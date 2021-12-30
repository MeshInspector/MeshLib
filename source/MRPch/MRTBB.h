#pragma once

// this is to include all important for us Intel Threading Building Blocks (TBB) parts in a precompiled header and suppress warnings there

#define TBB_SUPPRESS_DEPRECATED_MESSAGES 1
#pragma warning(push)
#pragma warning(disable: 4464) //relative include path contains '..'
#pragma warning(disable: 4574) //'TBB_USE_DEBUG' is defined to be '0': did you mean to use '#if TBB_USE_DEBUG'?
#pragma warning(disable: 5215) //a function parameter with a volatile qualified type is deprecated in C++20
#pragma warning(disable: 5219) //implicit conversion from '__int64' to 'double', possible loss of data
#pragma warning(disable: 5220) //a non-static data member with a volatile qualified type no longer implies that compiler generated copy/move constructors and copy/move assignment operators are not trivial
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>
#pragma warning(pop)
