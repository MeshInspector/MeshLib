#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "exports.h"
#include <vector>
#include <string>
#include <filesystem>

namespace MR
{

struct EndMillTool;

// class for storing CNC tools
class MRVIEWER_CLASS GcodeToolsLibrary
{
public:
    MRVIEWER_API GcodeToolsLibrary( const std::string& libraryName );

    // draw interface for interacting with storage (based on UI::combo)
    MRVIEWER_API bool drawInterface();

    // draw dialog for creating end mill tools
    // must be called outside any BeginCustomStatePlugin/EndCustomStatePlugin blocks
    MRVIEWER_API bool drawCreateToolDialog( float menuScaling );

    // get selected tool as ObjectMesh
    [[nodiscard]] MRVIEWER_API const std::shared_ptr<ObjectMesh>& getToolObject();

    // get selected tool as EndMillTool (if available)
    [[nodiscard]] const std::shared_ptr<EndMillTool>& getEndMillTool() const { return endMillTool_; }

    // set object specific size for automatic resize default object
    MRVIEWER_API void setAutoSize( float size );
private:

    // storage folder
    std::filesystem::path getFolder_();
    // get valid files in storage folder
    void updateFilesList_();
    // add new tool in storage from file with mesh
    void addNewToolFromFile_();
    // add new tool in storage from exist ObjectMesh
    void addNewToolFromMesh_( const std::shared_ptr<ObjectMesh>& objMesh );
    // add new end mill tool in storage
    void addNewTool_( const std::string& name, const EndMillTool& tool );
    // remove the selected tool from storage
    void removeSelectedTool_();
    // draw popup with ObjectMeshes available for adding
    void drawSelectMeshPopup_();
    // choose mesh from storage (and load it)
    bool loadFromFile_( const std::string& filename );

    std::string libraryName_;
    std::vector<std::string> filesList_;
    std::string selectedFileName_;
    std::shared_ptr<ObjectMesh> toolMesh_;
    std::shared_ptr<ObjectMesh> defaultToolMesh_; // default mesh
    std::shared_ptr<EndMillTool> endMillTool_;
    float autoSize_ = 0.f; // object specific size for automatic resize default object

    // Create Tool dialog state
    bool createToolDialogIsOpen_ = false;
    std::string createToolName_ = "Flat End Mill D2";
    int createToolType_ = 0;
    float createToolLength_ = 8.f;
    float createToolDiameter_ = 2.f;
    float createToolCornerRadius_ = 0.5f;
    float createToolCuttingAngle_ = 90.f;
    float createToolEndDiameter_ = 0.f;
};

}
