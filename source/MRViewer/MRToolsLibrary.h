#pragma once
#include <vector>
#include <string>
#include <filesystem>
#include "MRMesh/MRMeshFwd.h"
#include "exports.h"

namespace MR
{

class MRVIEWER_CLASS GcodeToolsLibrary
{
public:
    MRVIEWER_API GcodeToolsLibrary( const std::string& libraryName );

    MRVIEWER_API bool drawCombo();

    const std::shared_ptr<ObjectMesh>& getToolObject() { return toolMesh_; }
private:

    std::filesystem::path getFolder_();
    void updateFilesList_();
    void addNewToolFromFile_();
    void addNewToolFromMesh_( const std::shared_ptr<ObjectMesh>& objMesh );
    void drawSelectMeshPopup_();
    bool loadMeshFromFile_( const std::string& filename );

    std::string libraryName_;
    std::vector<std::string> filesList_;
    std::string selectedFileName_;
    std::shared_ptr<ObjectMesh> toolMesh_;
    std::shared_ptr<ObjectMesh> defaultToolMesh_;
};

}
