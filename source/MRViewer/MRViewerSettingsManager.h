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

    virtual void loadSettings( Viewer& ) = 0;
    virtual void saveSettings( const Viewer& ) = 0;
};

class MRVIEWER_CLASS ViewerSettingsManager : public IViewerSettingsManager
{
public:
    MRVIEWER_API ViewerSettingsManager();

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
    MRVIEWER_API int getLastExtentionNum( ObjType objType );
    MRVIEWER_API void setLastExtentionNum( ObjType objType, int num );
private:
    std::vector<int> lastExtentionNums_;
};
}
