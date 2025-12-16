#pragma once

#include "MRViewerFwd.h"
#include <string>
#include <vector>

namespace MR
{

// This class manages local user settings of viewer
// loading it when the app starts and saving it when it ends
class IViewerSettingsManager
{
public:
    virtual ~IViewerSettingsManager() = default;

    virtual int loadInt( const std::string& name, int def = 0 ) = 0;
    virtual void saveInt( const std::string& name, int value ) = 0;

    virtual std::string loadString( const std::string& name, const std::string& def = "" ) = 0;
    virtual void saveString( const std::string& name, const std::string& value ) = 0;

    virtual bool loadBool( const std::string& name, bool def = false ) = 0;
    virtual void saveBool( const std::string& name, bool value ) = 0;

    virtual void resetSettings( Viewer& ) = 0;
    virtual void loadSettings( Viewer& ) = 0;
    virtual void saveSettings( const Viewer& ) = 0;
};

class MRVIEWER_CLASS ViewerSettingsManager : public IViewerSettingsManager
{
public:
    MRVIEWER_API ViewerSettingsManager();

    MRVIEWER_API virtual int loadInt( const std::string& name, int def ) override;
    MRVIEWER_API virtual void saveInt( const std::string& name, int value ) override;
    MRVIEWER_API virtual std::string loadString( const std::string& name, const std::string& def ) override;
    MRVIEWER_API virtual void saveString( const std::string& name, const std::string& value ) override;
    MRVIEWER_API virtual bool loadBool( const std::string& name, bool def ) override;
    MRVIEWER_API virtual void saveBool( const std::string& name, bool value ) override;

    MRVIEWER_API virtual void resetSettings( Viewer& ) override;
    MRVIEWER_API virtual void loadSettings( Viewer& viewer ) override;
    MRVIEWER_API virtual void saveSettings( const Viewer& viewer ) override;

    enum class ObjType
    {
        Mesh = 0,
        Lines,
        Points,
        Voxels,
        DistanceMap,
        Count
    };
    MRVIEWER_API const std::string & getLastExtention( ObjType objType );
    MRVIEWER_API void setLastExtention( ObjType objType, std::string ext );
private:
    std::vector<std::string> lastExtentions_;
};
}
