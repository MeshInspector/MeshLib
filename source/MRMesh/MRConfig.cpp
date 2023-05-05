#include "MRConfig.h"
#include "MRStringConvert.h"
#include <assert.h>
#include <iostream>
#include <fstream>
#include "MRPch/MRSpdlog.h"
#include "MRSystem.h"
#include "MRPch/MRJson.h"

namespace MR
{

Config::Config()
{}

void Config::reset( std::string appName )
{
    if ( !filePath_.empty() )
    {
        writeToFile();
    }
    appName_ = std::move( appName );
    reset( getUserConfigFilePath() );
}

const std::string& Config::getAppName() const
{
    return appName_;
}

void Config::writeToFile()
{
    std::ofstream os( filePath_ );
    if ( loggerHandle_ )
        loggerHandle_->info( "Saving config file: " + utf8string( filePath_ ) );
    if ( os.is_open() )
    {
        os << config_;
        os.close();
    }
    else
    {
        if ( loggerHandle_ )
            loggerHandle_->warn( "Failed to save json config file " + utf8string( filePath_ ) );
    }
}

void Config::reset( const std::filesystem::path& filePath )
{
    if ( std::filesystem::exists( filePath ) )
    {
        auto readRes = deserializeJsonValue( filePath );
        if ( !readRes.has_value() )
        {
            if ( loggerHandle_ )
                loggerHandle_->error( readRes.error() );
        }
        else
        {
            config_ = std::move( readRes.value() );
        }
    }
    else
    {
        if ( loggerHandle_ )
            loggerHandle_->warn( "Failed to open json config file " + utf8string( Config::filePath_ ) );
    }
    filePath_ = filePath;
}

Config& Config::instance()
{
    static Config cfg;
    return cfg;
}

Config::~Config()
{
    writeToFile();
}

bool Config::hasBool( const std::string& key ) const
{
    return ( !config_[key].isNull() && config_[key].isBool() );
}
bool Config::getBool( const std::string& key, bool defaultValue ) const
{
    if ( !config_[key].isNull() )
    {
        return config_[key].asBool();
    }
    if ( loggerHandle_ )
        loggerHandle_->warn( "Key {} does not exist, default value \"{}\" returned", key, defaultValue );
    return defaultValue;
}
void Config::setBool( const std::string& key, bool keyValue )
{
    config_[key] = keyValue;
}

bool Config::hasColor( const std::string& key ) const
{
    return ( config_[key].isObject() &&
        config_[key]["r"].isNumeric() && config_[key]["g"].isNumeric() && config_[key]["b"].isNumeric() && config_[key]["a"].isNumeric() );
}
Color Config::getColor( const std::string& key, const Color& defaultValue ) const
{

    if ( config_[key].isObject() )
    {
        auto& val = config_[key];
        Color res;
        deserializeFromJson( val, res );
        return res;
    }
    if ( loggerHandle_ )
        loggerHandle_->warn( "Key {} does not exist, default value \"r:{} g:{} b:{} a:{}\" returned", key, 
            defaultValue.r, defaultValue.g, defaultValue.b, defaultValue.a );
    return defaultValue;
}
void Config::setColor( const std::string& key, const Color& keyValue )
{
    serializeToJson( keyValue, config_[key] );
}

bool Config::hasFileStack( const std::string& key ) const
{
    return ( config_[key].isArray() );
}
FileNamesStack Config::getFileStack( const std::string& key, const FileNamesStack& defaultValue ) const
{
    if ( config_[key].isArray() )
    {
        auto& val = config_[key];
        FileNamesStack res;
        for ( auto& v : val )
        {
            res.push_back( pathFromUtf8( v.asString() ) );
        }
        return res;
    }
    if ( loggerHandle_ )
        loggerHandle_->warn( "Key {} does not exist, default value returned", key );
    return defaultValue;
}
void Config::setFileStack( const std::string& key, const FileNamesStack& keyValue )
{
    for ( auto i = 0; i < keyValue.size(); i++ )
    {
        config_[key][i] = utf8string( keyValue[i] );
    }
}

bool Config::hasVector2i( const std::string& key ) const
{
    return config_[key].isObject() && ( config_[key]["x"].isInt() && config_[key]["y"].isInt() );
}

Vector2i Config::getVector2i( const std::string& key, const Vector2i& defaultValue /*= Vector2i( 0, 0 ) */ ) const
{
    auto& val = config_[key];
    Vector2i res = defaultValue;
    deserializeFromJson( val, res );
    return res;
}

void Config::setVector2i( const std::string& key, const Vector2i& keyValue )
{
    serializeToJson( keyValue, config_[key] );
}

bool Config::hasJsonValue( const std::string& key )
{
    return config_.isMember( key );
}

Json::Value Config::getJsonValue( const std::string& key, const Json::Value& defaultValue /*= {} */ )
{
    if ( hasJsonValue( key ) )
        return config_[key];
    return defaultValue;
}

void Config::setJsonValue( const std::string& key, const Json::Value& keyValue )
{
    config_[key] = keyValue;
}

} //namespace MR
