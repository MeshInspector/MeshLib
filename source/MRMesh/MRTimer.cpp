#include "MRTimer.h"
#include "MRLog.h"
#include <map>
#include <thread>
#include <sstream>

using namespace std::chrono;

namespace MR
{

struct TimeRecord
{
    int count = 0;
    nanoseconds time = {};
    std::map<std::string, TimeRecord> children;

    // returns summed time of immediate children
    nanoseconds childTime() const;

    double seconds() const { return time.count() * 1e-9; }
    double mySeconds() const { return ( time - childTime() ).count() * 1e-9; }
};

nanoseconds TimeRecord::childTime() const
{
    auto res = nanoseconds{ 0 };
    for ( const auto& child : children )
        res += child.second.time;
    return res;
}

void printTimeRecord( const TimeRecord& timeRecord, const std::string& name, int indent, const std::shared_ptr<spdlog::logger>& loggerHandle )
{
    std::stringstream ss;
    ss << std::setw( 9 )  << std::right << timeRecord.count;
    ss << std::setw( 12 ) << std::right << std::fixed << std::setprecision( 3 ) << timeRecord.seconds();
    ss << std::setw( 12 ) << std::right << std::fixed << std::setprecision( 3 ) << timeRecord.mySeconds();
    ss << std::string( indent, ' ' ) << name;
    loggerHandle->info( ss.str() );

    for ( const auto& child : timeRecord.children )
        printTimeRecord( child.second, child.first, indent + 4, loggerHandle );
}

struct RootTimeRecord : TimeRecord
{
    time_point<high_resolution_clock> started = high_resolution_clock::now();
    bool printTreeInDtor = true;
    // prolong logger life
    std::shared_ptr<spdlog::logger> loggerHandle = Logger::instance().getSpdLogger();
    RootTimeRecord()
    {
        count = 1;
    }
    void printTree()
    {
        loggerHandle->info( "Time Tree:" );
        std::stringstream ss;
        ss << std::setw( 9 ) << std::right << "Count";
        ss << std::setw( 12 ) << std::right << "Time";
        ss << std::setw( 12 ) << std::right << "Self time";
        ss << "    Name";
        loggerHandle->info( ss.str() );
        time = high_resolution_clock::now() - started;
        printTimeRecord( *this, "(total)", 4, loggerHandle );
    }
    ~RootTimeRecord()
    {
        if ( !printTreeInDtor )
            return;
        printTree();
    }
};

static RootTimeRecord rootTimeRecord;
static TimeRecord* currentRecord = &rootTimeRecord;
static auto mainThreadId = std::this_thread::get_id();

void printTimingTreeAtEnd( bool on )
{
    rootTimeRecord.printTreeInDtor = on;
}

void printTimingTreeAndStop()
{
    rootTimeRecord.printTree();
    printTimingTreeAtEnd( false );
}

void Timer::restart( std::string name )
{
    finish();
    if ( std::this_thread::get_id() != mainThreadId )
        return;
    name_ = std::move( name );
    start_ = high_resolution_clock::now();
    parent_ = currentRecord;
    currentRecord = &currentRecord->children[name_];
}

void Timer::finish()
{
    if ( !parent_ )
        return;

    currentRecord->time += high_resolution_clock::now() - start_;
    ++currentRecord->count;
    currentRecord = parent_;
    parent_ = nullptr;
}

} //namespace MR
