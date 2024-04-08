#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "MRViewer/MRStatePlugin.h"
#include "MRViewer/MRRibbonMenuItem.h"
#include "MRMesh/MRIOFilters.h"
#include <filesystem>

namespace MR
{
using FileNamesStack = std::vector<std::filesystem::path>;

class OpenDirectoryMenuItem : public RibbonMenuItem
{
public:
    OpenDirectoryMenuItem();
    virtual bool action() override;
    void openDirectory( const std::filesystem::path& directory ) const;
};

class OpenFilesMenuItem : public RibbonMenuItem, public MultiListener<DragDropListener>
{
public:
    OpenFilesMenuItem();
    ~OpenFilesMenuItem();
    virtual bool action() override;

    virtual const DropItemsList& dropItems() const override;
private:
    virtual bool dragDrop_( const std::vector<std::filesystem::path>& paths ) override;
    void parseLaunchParams_();
    void setupListUpdate_();
    bool checkPaths_( const std::vector<std::filesystem::path>& paths );

    boost::signals2::scoped_connection recentStoreConnection_;
    FileNamesStack recentPathsCache_;
    IOFilters filters_;
    std::shared_ptr<OpenDirectoryMenuItem> openDirectoryItem_;
};

#ifndef MRMESH_NO_DICOM
class OpenDICOMsMenuItem : public RibbonMenuItem
{
public:
    OpenDICOMsMenuItem();
    virtual bool action() override;
};
#endif

class SaveObjectMenuItem : public RibbonMenuItem
{
public:
    SaveObjectMenuItem();
    virtual bool action() override;
    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>&objs ) const override;
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

class CaptureScreenshotMenuItem : public StatePlugin
{
public:
    CaptureScreenshotMenuItem();
    virtual void drawDialog( float menuScaling, ImGuiContext* ) override;
    virtual bool blocking() const override { return false; }
private:
    Vector2i resolution_;
    bool transparentBg_{ true };
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