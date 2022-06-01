#pragma once
#include "MRMeshFwd.h"
#include "MRColor.h"
#include "MRSerializer.h"
#include "MRPch/MRJson.h"
#include "MRVector2.h"
#include "MRLog.h"
#include <filesystem>
#include <string>

namespace MR
{

using FileNamesStack = std::vector<std::filesystem::path>;

class Config
{
public:
    Config( Config const& ) = delete;
    void operator=( Config const& ) = delete;

    MRMESH_API static Config& instance();

    // looks up (~/.local/share/<appname>/config.json) or (AppData\<appname>\config.json)
    // creates directory if not presented
    MRMESH_API void reset( const std::string& appName );

private:
    Config();

    // looks for presented *.json file
    void reset( const std::filesystem::path& filePath );

    // writes current config to file
    void writeToFile();

    // stores configuration depends on constructor call: (<filepath>) or
    // (~/.local/share/<appname>/config.json) or (AppData\<appname>\config.json)
    ~Config();

public:
    // returns true if MRColor with presented key exists
    MRMESH_API bool hasBool( const std::string& key ) const;
    // returns MRColor with presented key
    MRMESH_API bool getBool( const std::string& key, bool defaultValue = false ) const;
    // sets MRColor for presented key
    MRMESH_API void setBool( const std::string& key, bool keyValue );

    // returns true if MRColor with presented key exists
    MRMESH_API bool hasColor( const std::string& key ) const;
    // returns MRColor with presented key
    MRMESH_API Color getColor( const std::string& key, const Color& defaultValue = Color::black() ) const;
    // sets MRColor for presented key
    MRMESH_API void setColor( const std::string& key, const Color& keyValue );

    // returns true if 'recently used' files exist
    MRMESH_API bool hasFileStack( const std::string& key ) const;
    // returns 'recently used' files list
    MRMESH_API FileNamesStack getFileStack( const std::string& key, const FileNamesStack& defaultValue = FileNamesStack() ) const;
    // sets 'recently used' files list
    MRMESH_API void setFileStack( const std::string& key, const FileNamesStack& keyValue );

    // returns true if Vector2i with presented key exists
    MRMESH_API bool hasVector2i( const std::string& key ) const;
    // returns Vector2i with presented key
    MRMESH_API Vector2i getVector2i( const std::string& key, const Vector2i& defaultValue = Vector2i() ) const;
    // sets Vector2i for presented key
    MRMESH_API void setVector2i( const std::string& key, const Vector2i& keyValue );

    // returns true if json value with this key exists
    MRMESH_API bool hasJsonValue( const std::string& key );
    // returns custom json value
    MRMESH_API Json::Value getJsonValue( const std::string& key, const Json::Value& defaultValue = {} );
    // sets custom json value
    MRMESH_API void setJsonValue( const std::string& key, const Json::Value& keyValue );

private:
    std::string appName_;

    Json::Value config_;
    std::filesystem::path filePath_;
    // prolong logger life
    std::shared_ptr<spdlog::logger> loggerHandle_ = Logger::instance().getSpdLogger();
};

// returns the list of libraries with plugins for application. See Json file from the executable directory for details.
MRMESH_API std::optional<std::vector<std::filesystem::path>> getPluginLibraryList();

} // namespace MR
