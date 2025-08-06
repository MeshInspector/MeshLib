#include "MRTimer.h"
#include "MRTimeRecord.h"
#include "MRPch/MRSpdlog.h"

#include <sstream>

using namespace std::chrono;

namespace MR
{

nanoseconds TimeRecord::childTime() const
{
    auto res = nanoseconds{ 0 };
    for ( const auto& child : children )
        res += child.second.time;
    return res;
}

void printTimeRecord( const TimeRecord& timeRecord, const std::string& name, int indent, const std::shared_ptr<spdlog::logger>& loggerHandle, double minTimeSec )
{
    auto sec = timeRecord.seconds();
    if ( sec < minTimeSec )
        return;

    std::stringstream ss;
    ss << std::setw( 9 )  << std::right << timeRecord.count;
    ss << std::setw( 12 ) << std::right << std::fixed << std::setprecision( 3 ) << sec;
    ss << std::setw( 12 ) << std::right << std::fixed << std::setprecision( 3 ) << timeRecord.mySeconds();
    ss << std::string( indent, ' ' ) << name;
    loggerHandle->info( ss.str() );

    for ( const auto& child : timeRecord.children )
        printTimeRecord( child.second, child.first, indent + 4, loggerHandle, minTimeSec );
}

void summarizeRecords( const TimeRecord& root, const std::string& name, std::map<std::string, SimpleTimeRecord> & res )
{
    auto & x = res[name];
    x.count += root.count;
    x.time += root.myTime();

    for ( const auto& child : root.children )
        summarizeRecords( child.second, child.first, res );
}

void printSummarizedRecords( const TimeRecord& root, const std::string& name, const std::shared_ptr<spdlog::logger>& loggerHandle, double minTimeSec )
{
    std::map<std::string, SimpleTimeRecord> sum;
    summarizeRecords( root, name, sum );

    std::vector<std::pair<std::string, SimpleTimeRecord>> sumPairs;
    sumPairs.reserve( sum.size() );
    for ( const auto & p : sum )
        sumPairs.push_back( p );
    // sort in time-descending order
    std::sort( sumPairs.begin(), sumPairs.end(), []( const auto & a, const auto & b )
        { return a.second.time > b.second.time; } );

    loggerHandle->info( "" );
    loggerHandle->info( "Slowest places:" );
    std::stringstream ss;
    ss << std::setw( 9 ) << std::right << "Count";
    ss << std::setw( 12 ) << std::right << "Self time";
    ss << "    Name";
    loggerHandle->info( ss.str() );

    int skipCount = 0;
    double skipSec = 0;
    for ( const auto & p : sumPairs )
    {
        auto sec = p.second.seconds();
        if ( sec < minTimeSec )
        {
            skipCount += p.second.count;
            skipSec += sec;
            continue;
        }
        ss = std::stringstream{};
        ss << std::setw( 9 )  << std::right << p.second.count;
        ss << std::setw( 12 ) << std::right << std::fixed << std::setprecision( 3 ) << sec;
        ss << "    " << p.first;
        loggerHandle->info( ss.str() );
    }
    if ( skipCount > 0 )
    {
        ss = std::stringstream{};
        ss << std::setw( 9 )  << std::right << skipCount;
        ss << std::setw( 12 ) << std::right << std::fixed << std::setprecision( 3 ) << skipSec;
        ss << std::defaultfloat << "    (others, each faster than " << minTimeSec << " sec)";
        loggerHandle->info( ss.str() );
    }
}

static thread_local TimeRecord* currentRecord = nullptr;

ThreadRootTimeRecord::ThreadRootTimeRecord( const char * tdName ) : threadName( tdName )
{
    count = 1;
}

void ThreadRootTimeRecord::printTree()
{
    loggerHandle->info( "{} thread time tree (min printed time {} sec):", threadName, minTimeSec );
    std::stringstream ss;
    ss << std::setw( 9 ) << std::right << "Count";
    ss << std::setw( 12 ) << std::right << "Time";
    ss << std::setw( 12 ) << std::right << "Self time";
    ss << "    Name";
    loggerHandle->info( ss.str() );
    time = high_resolution_clock::now() - started;
    printTimeRecord( *this, "(total)", 4, loggerHandle, minTimeSec );
    printSummarizedRecords( *this, "(not covered by timers)", loggerHandle, minTimeSec );
}

ThreadRootTimeRecord::~ThreadRootTimeRecord()
{
    if ( !printTreeInDtor )
        return;
    printTree();
}

void registerThreadRootTimeRecord( ThreadRootTimeRecord & root )
{
    if( currentRecord == nullptr )
        currentRecord = &root;
    else
        assert( false );
}

void unregisterThreadRootTimeRecord( ThreadRootTimeRecord & root )
{
    if( currentRecord == &root )
        currentRecord = nullptr;
    else
        assert( false );
}

static struct MainThreadRootTimeRecord  : public ThreadRootTimeRecord
{
    MainThreadRootTimeRecord() : ThreadRootTimeRecord( "Main" )
    {
        assert( currentRecord == nullptr );
        currentRecord = this;
    }
} rootTimeRecord;

void printTimingTreeAtEnd( bool on, double minTimeSec )
{
    rootTimeRecord.printTreeInDtor = on;
    rootTimeRecord.minTimeSec = minTimeSec;
}

void printCurrentTimerBranch()
{
    Timer t( "Print Timer branch leaf" );
    const TimeRecord* active = currentRecord;
    auto& logger = rootTimeRecord.loggerHandle;
    if ( !logger )
        return;
    while ( active )
    {
        if ( !active->parent )
        {
            logger->info( "Root" );
            break;
        }
        for ( const auto& child : active->parent->children )
        {
            if ( &child.second != active )
                continue;
            logger->info( child.first );
            break;
        }
        active = active->parent;
    }
}

void printTimingTree( double minTimeSec )
{
    rootTimeRecord.minTimeSec = minTimeSec;
    rootTimeRecord.printTree();
}

void Timer::restart( std::string name )
{
    finish();
    start( std::move( name ) );
}

void Timer::start( std::string name )
{
    auto parent = currentRecord;
    if ( !parent )
        return;
    started_ = true;
    start_ = high_resolution_clock::now();
    currentRecord = &parent->children[ std::move( name ) ];
    currentRecord->parent = parent;
}

void Timer::finish()
{
    if ( !started_ )
        return;
    started_ = false;
    auto currentParent = currentRecord->parent;
    if ( !currentParent )
        return;

    currentRecord->time += high_resolution_clock::now() - start_;
    ++currentRecord->count;

    currentRecord = currentParent;
}

} //namespace MR
