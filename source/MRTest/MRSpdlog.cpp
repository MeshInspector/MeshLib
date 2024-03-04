#include <MRMesh/MRGTest.h>
#include <MRPch/MRSpdlog.h>

TEST( MRPch, Spdlog )
{
    // check for correct type casts on macOS
    auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto sink = std::dynamic_pointer_cast<spdlog::sinks::sink>( consoleSink );
    auto castedSink = std::dynamic_pointer_cast<spdlog::sinks::stdout_color_sink_mt>( sink );
    EXPECT_TRUE( castedSink );
}
