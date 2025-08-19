#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "MRViewer/MRStatePlugin.h"
#include "MRViewer/MRRibbonMenuItem.h"
#include "MRMesh/MRIOFilters.h"
#include <filesystem>

#ifndef MESHLIB_NO_VOXELS
#include "MRVoxels/MRVoxelsFwd.h"
#endif

namespace MR
{
using FileNamesStack = std::vector<std::filesystem::path>;

class OpenDirectoryMenuItem : public RibbonMenuItem
{
public:
    OpenDirectoryMenuItem();
    std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const override;
    bool action() override;
    void openDirectory( const std::filesystem::path& directory ) const;
};

class OpenFilesMenuItem : public RibbonMenuItem, public MultiListener<DragEntranceListener, DragOverListener, DragDropListener, PreDrawListener>
{
public:
    OpenFilesMenuItem();
    ~OpenFilesMenuItem();
    virtual bool action() override;

    virtual const DropItemsList& dropItems() const override;
private:
    virtual void dragEntrance_( bool entered ) override;
    virtual bool dragOver_( int x, int y ) override;
    virtual bool dragDrop_( const std::vector<std::filesystem::path>& paths ) override;

    virtual void preDraw_() override;

    bool dragging_{ false };
    Vector2i dragPos_;

    void parseLaunchParams_();
    void setupListUpdate_();

    boost::signals2::scoped_connection recentStoreConnection_;
    FileNamesStack recentPathsCache_;
    IOFilters filters_;
    std::shared_ptr<OpenDirectoryMenuItem> openDirectoryItem_;
};

#if !defined( MESHLIB_NO_VOXELS ) && !defined( MRVOXELS_NO_DICOM )
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
    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const override;

protected:
    void saveSceneAs_();
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