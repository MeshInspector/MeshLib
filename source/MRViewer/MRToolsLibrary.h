#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "exports.h"
#include <vector>
#include <string>
#include <filesystem>

namespace MR
{

// class for storing CNC tools
class MRVIEWER_CLASS GcodeToolsLibrary
{
public:
    MRVIEWER_API GcodeToolsLibrary( const std::string& libraryName );

    // draw interface for interacting with storage (based on UI::combo)
    MRVIEWER_API bool drawInterface();

    // get selected tool as ObjectMesh
    MRVIEWER_API const std::shared_ptr<ObjectMesh>& getToolObject();

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
    // draw popup with ObjectMeshes available for adding
    void drawSelectMeshPopup_();
    // choose mesh from storage (and load it)
    bool loadMeshFromFile_( const std::string& filename );

    std::string libraryName_;
    std::vector<std::string> filesList_;
    std::string selectedFileName_;
    std::shared_ptr<ObjectMesh> toolMesh_;
    std::shared_ptr<ObjectMesh> defaultToolMesh_; // default mesh
    float autoSize_ = 0.f; // object specific size for automatic resize default object
};

}
