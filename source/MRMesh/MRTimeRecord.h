#pragma once

#include "MRMeshFwd.h"
#include "MRLog.h"
#include "MRPch/MRBindingMacros.h"
#include <chrono>
#include <map>
#include <string>

namespace MR
{

/// \addtogroup BasicGroup
/// \{

struct MR_BIND_IGNORE SimpleTimeRecord
{
    int count = 0;
    std::chrono::nanoseconds time = {};
    double seconds() const { return time.count() * 1e-9; }
};

struct MR_BIND_IGNORE TimeRecord : SimpleTimeRecord
{
    TimeRecord* parent = nullptr;
    std::map<std::string, TimeRecord> children;

    // returns summed time of immediate children
    MRMESH_API std::chrono::nanoseconds childTime() const;
    std::chrono::nanoseconds myTime() const { return time - childTime(); }

    double mySeconds() const { return myTime().count() * 1e-9; }
};

struct MR_BIND_IGNORE ThreadRootTimeRecord : TimeRecord
{
    const char * threadName = nullptr;
    std::chrono::time_point<std::chrono::high_resolution_clock> started = std::chrono::high_resolution_clock::now();
    bool printTreeInDtor = true;
    double minTimeSec = 0.1;
    // prolong logger life
    std::shared_ptr<spdlog::logger> loggerHandle = Logger::instance().getSpdLogger();
    MRMESH_API ThreadRootTimeRecord( const char * tdName );
    MRMESH_API void printTree();
    MRMESH_API ~ThreadRootTimeRecord();
};

/// installs given record in the current thread (no record must be installed before)
MR_BIND_IGNORE MRMESH_API void registerThreadRootTimeRecord( ThreadRootTimeRecord & root );

/// un-installs given record in the current thread
MR_BIND_IGNORE MRMESH_API void unregisterThreadRootTimeRecord( ThreadRootTimeRecord & root );

/// \}

} // namespace MR
