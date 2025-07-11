#include "MRConfig.h"
#include "MRStringConvert.h"
#include "MRSystem.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRWasm.h"

#include <cassert>
#include <iostream>
#include <fstream>

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
#ifndef __EMSCRIPTEN__
    // although json is a textual format, we open the file in binary mode to get exactly the same result on Windows and Linux
    std::ofstream os( filePath_, std::ofstream::binary );
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
#else
    std::stringstream strStream;
    strStream << config_;
    std::string str = strStream.str();
#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wdollar-in-identifier-extension"
    EM_ASM({ localStorage.setItem( 'config', UTF8ToString( $0 ) ) }, str.c_str() );
#pragma GCC diagnostic pop
#endif
}

void Config::reset( const std::filesystem::path& filePath )
{
#ifndef __EMSCRIPTEN__
    std::error_code ec;
    if ( std::filesystem::exists( filePath, ec ) )
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
#else
    auto *jsStr = (char *)EM_ASM_PTR({
        var configStr = localStorage.getItem('config');
        if ( configStr == null )
            configStr = "";
        var lengthBytes = lengthBytesUTF8( configStr ) + 1;
        var stringOnWasmHeap = _malloc( lengthBytes );
        stringToUTF8( configStr, stringOnWasmHeap, lengthBytes );
        return stringOnWasmHeap;
    });
    std::string configStr;
    if ( jsStr )
    {
        configStr = std::string( jsStr );
        free(jsStr);
    }
    if (!configStr.empty())
    {
        auto readRes = deserializeJsonValue( configStr );
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
            loggerHandle_->warn( "Failed to load config from localStorage" + utf8string( Config::filePath_ ) );
    }

#endif
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
        loggerHandle_->debug( "Key {} does not exist, default value \"{}\" returned", key, defaultValue );
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
        loggerHandle_->debug( "Key {} does not exist, default value \"r:{} g:{} b:{} a:{}\" returned", key, 
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
        loggerHandle_->debug( "Key {} does not exist, default value returned", key );
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

bool MR::Config::hasEnum( const Enum& enumeration, const std::string& key ) const
{
    if ( !config_[key].isString() )
        return false;
    std::string value = config_[key].asString();
    for ( const char* e : enumeration )
        if ( value == e )
            return true;
    return false;
}

int MR::Config::getEnum( const Enum& enumeration, const std::string& key, int defaultValue ) const
{
    if ( !config_[key].isString() )
        return defaultValue;
    std::string value = config_[key].asString();
    for ( size_t i = 0; i < enumeration.size(); i++ )
        if ( value == enumeration[i] )
            return ( int )i;
    return defaultValue;
}

void MR::Config::setEnum( const Enum& enumeration, const std::string& key, int keyValue )
{
    config_[key] = enumeration[keyValue];
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
