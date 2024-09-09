#include "MRToolsLibrary.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystem.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshSave.h"
#include "MRMesh/MRMeshLoad.h"
#include "MRFileDialog.h"
#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRCylinder.h"
#include "MRUIStyle.h"
#include "MRRibbonConstants.h"
#include "MRMesh/MRDirectory.h"
#include <imgui.h>
#include <cassert>

namespace MR
{

const char defaultName[] = "Default";

GcodeToolsLibrary::GcodeToolsLibrary( const std::string& libraryName )
{
    assert( !libraryName.empty() );
    libraryName_ = libraryName;

    selectedFileName_ = defaultName;
}

bool GcodeToolsLibrary::drawInterface()
{
    bool result = false;

    if ( UI::beginCombo( "Tool Mesh", selectedFileName_ ) )
    {
        bool selected = selectedFileName_ == defaultName;
        if ( ImGui::Selectable( defaultName, &selected ) )
        {
            toolMesh_ = defaultToolMesh_;
            selectedFileName_ = defaultName;
            result = true;
        }

        updateFilesList_();
        for ( int i = 0; i < filesList_.size(); ++i )
        {
            selected = selectedFileName_ == filesList_[i];

            if ( ImGui::Selectable( filesList_[i].c_str(), &selected ) && selected )
            {
                result = loadMeshFromFile_( filesList_[i] );
            }
        }
        

        if ( !getFolder_().empty() )
        {
            selected = false;
            if ( ImGui::Selectable( "<New Tool from File>", &selected ) )
            {
                addNewToolFromFile_();
                result = true;
            }

            const bool anyMeshExist = bool( getDepthFirstObject<ObjectMesh>( &SceneRoot::get(), ObjectSelectivityType::Selectable ) );
            if ( !anyMeshExist )
            {
                ImGui::PushStyleColor( ImGuiCol_Text, ImGui::GetStyleColorVec4( ImGuiCol_TextDisabled ) );
                ImGui::Text( "%s", "<New Tool from exist Mesh>" );
                ImGui::PopStyleColor();
            } else if ( ImGui::BeginMenu( "<New Tool from exist Mesh>" ) )
            {
                auto objsMesh = getAllObjectsInTree<ObjectMesh>( &SceneRoot::get(), ObjectSelectivityType::Selectable );
                for ( int i = 0; i < objsMesh.size(); ++i )
                {
                    selected = false;
                    if ( ImGui::Selectable( objsMesh[i]->name().c_str(), &selected ) )
                    {
                        result = true;
                        addNewToolFromMesh_( objsMesh[i] );
                    }
                }

                ImGui::EndMenu();
            }
        }
        UI::endCombo();
    }
    const float btnWidth = ImGui::CalcTextSize( "Remove" ).x + ImGui::GetStyle().FramePadding.x * 2.f;
    const float btnHeight = ImGui::GetTextLineHeight() + StyleConsts::CustomCombo::framePadding.y * 2.f;
    const float btnPosX = ImGui::GetContentRegionAvail().x - btnWidth;
    
    ImGui::SameLine( btnPosX );
    if ( UI::button( "Remove", selectedFileName_ != defaultName, {btnWidth, btnHeight}) )
    {
        const auto folderPath = getFolder_();
        if ( !folderPath.empty() )
        {
            std::error_code ec;
            std::filesystem::remove( folderPath / ( selectedFileName_ + ".mrmesh" ), ec );
            selectedFileName_ = defaultName;
            result = true;
            toolMesh_ = defaultToolMesh_;
        }
    }

    return result;
}

const std::shared_ptr<MR::ObjectMesh>& GcodeToolsLibrary::getToolObject()
{
    if ( selectedFileName_ == defaultName )
    {
        if ( !defaultToolMesh_ )
        {
            defaultToolMesh_ = std::make_shared<ObjectMesh>();
            defaultToolMesh_->setName( "DefaultToolMesh" );
            const float autoSize = autoSize_ > 0.f ? autoSize_ : 100.f;
            auto meshPtr = std::make_shared<Mesh>( makeCylinder( 0.01f * autoSize, 0.08f * autoSize, 50 ) );
            defaultToolMesh_->setMesh( meshPtr );
        }

        if ( toolMesh_ != defaultToolMesh_ )
            toolMesh_ = defaultToolMesh_;
    }
    return toolMesh_;
}

void GcodeToolsLibrary::setAutoSize( float size )
{
    if ( size == autoSize_ || size <= 0.f )
        return;
    defaultToolMesh_.reset();
    autoSize_ = size;
}

std::filesystem::path GcodeToolsLibrary::getFolder_()
{
    const std::filesystem::path path = getUserConfigDir() / libraryName_;

    std::error_code ec;
    if ( std::filesystem::exists( path, ec ) )
        return path;
    else if ( std::filesystem::create_directory( path, ec ) )
        return path;
    
    return {};
}

void GcodeToolsLibrary::updateFilesList_()
{
    filesList_.clear();
    const auto folderPath = getFolder_();
    if ( folderPath.empty() )
        return;

    std::error_code ec;
    for ( auto entry : Directory{ folderPath, ec } )
    {
        if ( !entry.is_regular_file( ec ) )
            continue;
        const auto filename = entry.path().filename();
        if ( utf8string( filename.extension() ) == ".mrmesh" )
            filesList_.push_back( utf8string( filename.stem() ) );
    }
}

void GcodeToolsLibrary::addNewToolFromFile_()
{
    const auto folderPath = getFolder_();
    if ( folderPath.empty() )
        return;

    auto path = openFileDialog( { .filters = MeshLoad::getFilters() } );
    if ( path.empty() )
        return;

    auto loadRes = MeshLoad::fromAnySupportedFormat( path );
    if ( !loadRes )
        return;

    toolMesh_ = std::make_shared<ObjectMesh>();
    toolMesh_->setName( utf8string( path.filename().stem() ) );
    toolMesh_->setMesh( std::make_shared<Mesh>( *loadRes ) );
    MeshSave::toMrmesh( *loadRes, folderPath / ( toolMesh_->name() + ".mrmesh" ) );
    selectedFileName_ = toolMesh_->name();
}

void GcodeToolsLibrary::addNewToolFromMesh_( const std::shared_ptr<ObjectMesh>& objMesh )
{
    const auto folderPath = getFolder_();
    if ( folderPath.empty() )
        return;
    toolMesh_ = std::dynamic_pointer_cast< ObjectMesh >( objMesh->clone() );
    MeshSave::toMrmesh( *toolMesh_->mesh(), folderPath / ( toolMesh_->name() + ".mrmesh" ) );
    selectedFileName_ = toolMesh_->name();
}

void GcodeToolsLibrary::drawSelectMeshPopup_()
{
    if ( !ImGui::BeginPopup( "SelectMesh" ) )
        return;

    auto objsMesh = getAllObjectsInTree<ObjectMesh>( &SceneRoot::get(), ObjectSelectivityType::Selectable );
    for ( int i = 0; i < objsMesh.size(); ++i )
    {
        bool selected = false;
        if ( ImGui::Selectable( objsMesh[i]->name().c_str(), &selected ) )
            addNewToolFromMesh_( objsMesh[i] );
    }
    ImGui::EndPopup();
}

bool GcodeToolsLibrary::loadMeshFromFile_( const std::string& filename )
{
    const auto folderPath = getFolder_();
    if ( folderPath.empty() )
        return false;

    const auto path = folderPath / ( filename + ".mrmesh" );
    std::error_code ec;
    if ( !std::filesystem::exists( path, ec ) )
        return false;

    auto loadRes = MeshLoad::fromMrmesh( path );
    if ( !loadRes )
        return false;

    toolMesh_ = std::make_shared<ObjectMesh>();
    toolMesh_->setName( filename );
    toolMesh_->setMesh( std::make_shared<Mesh>( *loadRes ) );
    selectedFileName_ = filename;
    return true;
}

}
