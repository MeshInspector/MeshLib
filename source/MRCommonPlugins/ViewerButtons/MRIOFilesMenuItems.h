#pragma once
#include "MRViewer/MRRibbonMenuItem.h"
#include "MRMesh/MRIOFilters.h"

namespace MR
{
using FileNamesStack = std::vector<std::filesystem::path>;

class OpenFilesMenuItem : public RibbonMenuItem, public MultiListener<DragDropListener>
{
public:
    OpenFilesMenuItem();
    ~OpenFilesMenuItem();
    virtual bool action() override;

    virtual const DropItemsList& dropItems() const override;
private:
    virtual bool dragDrop_( const std::vector<std::filesystem::path>& paths ) override;
    void setupListUpdate_();
    void loadFiles_( const std::vector<std::filesystem::path>& paths );
    bool checkPaths_( const std::vector<std::filesystem::path>& paths );

    boost::signals2::scoped_connection recentStoreConnection_;
    FileNamesStack recentPathsCache_;
    IOFilters filters_;
};

class OpenDirectoryMenuItem : public RibbonMenuItem
{
public:
    OpenDirectoryMenuItem();
    virtual bool action() override;
};

class OpenDICOMsMenuItem : public RibbonMenuItem
{
public:
    OpenDICOMsMenuItem();
    virtual bool action() override;
};

class SaveObjectMenuItem : public RibbonMenuItem, public SceneStateOrCheck< SceneStateExactCheck<1, ObjectMesh>, SceneStateExactCheck<1, ObjectLines>, SceneStateExactCheck<1, ObjectPoints>, SceneStateExactCheck<1, ObjectVoxels> >
{
public:
    SaveObjectMenuItem();
    virtual bool action() override;
};

class SaveSelectedMenuItem : public RibbonMenuItem, public SceneStateAtLeastCheck<1, Object>
{
public:
    SaveSelectedMenuItem();
    virtual bool action() override;
};

class SaveSceneAsMenuItem : public RibbonMenuItem
{
public:
    SaveSceneAsMenuItem( const std::string& pluginName = "Save Scene As" );
    virtual bool action() override;

protected:
    void saveScene_( const std::filesystem::path& savePath );
};

class SaveSceneMenuItem : public SaveSceneAsMenuItem
{
public:
    SaveSceneMenuItem();
    virtual bool action() override;
};

class CaptureScreenshotMenuItem : public RibbonMenuItem
{
public:
    CaptureScreenshotMenuItem();
    virtual bool action() override;
};

class CaptureUIScreenshotMenuItem : public RibbonMenuItem
{
public:
    CaptureUIScreenshotMenuItem();
    virtual bool action() override;
};

class CaptureScreenshotToClipBoardMenuItem : public RibbonMenuItem
{
public:
    CaptureScreenshotToClipBoardMenuItem();
    virtual bool action() override;
    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const override;
};

}