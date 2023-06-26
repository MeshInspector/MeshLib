#include "MRToolsLibrary.h"
#include <cassert>
#include <imgui.h>
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

namespace MR
{

const char defaultName[] = "Default";

GcodeToolsLibrary::GcodeToolsLibrary( const std::string& libraryName )
{
    assert( !libraryName.empty() );
    libraryName_ = libraryName;
    if ( !std::filesystem::exists( getFolder_() ) )
        std::filesystem::create_directory( getFolder_() );
    
    defaultToolMesh_ = std::make_shared<ObjectMesh>();
    defaultToolMesh_->setName( "DefaultToolMesh" );
    auto meshPtr = std::make_shared<Mesh>( std::move( makeCylinder( 1.f, 8.f, 50 ) ) );
    defaultToolMesh_->setMesh( meshPtr );

    toolMesh_ = defaultToolMesh_;
    selectedFileName_ = defaultName;
}

bool GcodeToolsLibrary::drawCombo()
{
    bool openSelectMeshPopup = false;

    bool result = false;
    if ( ImGui::BeginCombo( "Tool Mesh", selectedFileName_.c_str() ) )
    {
        bool selected = selectedFileName_ == defaultName;
        if ( ImGui::Selectable( defaultName, &selected ) )
        {
            toolMesh_ = defaultToolMesh_;
            selectedFileName_ = defaultName;
            result = true;
            ImGui::CloseCurrentPopup();
        }

        updateFilesList_();
        for ( int i = 0; i < filesList_.size(); ++i )
        {
            selected = selectedFileName_ == filesList_[i];

            if ( ImGui::Selectable( filesList_[i].c_str(), &selected ) && selected )
            {
                result = loadMeshFromFile_( filesList_[i] );
                ImGui::CloseCurrentPopup();
            }
        }

        selected = false;
        if ( ImGui::Selectable( "<New Tool from File>", &selected ) )
        {
            addNewToolFromFile_();
            result = true;
            ImGui::CloseCurrentPopup();
        }

        const bool anyMeshExist = bool( getDepthFirstObject<ObjectMesh>( &SceneRoot::get(), ObjectSelectivityType::Selectable ) );
        if ( !anyMeshExist )
            ImGui::PushStyleColor( ImGuiCol_Text, ImGui::GetStyleColorVec4( ImGuiCol_TextDisabled ) );
        if ( ImGui::Selectable( "<New Tool from exist Mesh>", &selected ) && anyMeshExist )
        {
            openSelectMeshPopup = true;
            result = true;
            ImGui::CloseCurrentPopup();
        }
        if ( !anyMeshExist )
            ImGui::PopStyleColor();

        ImGui::EndCombo();
    }
    const float btnWidth = ImGui::CalcTextSize( "Remove" ).x + ImGui::GetStyle().FramePadding.x * 2.f;
    const float btnHeight = ImGui::GetTextLineHeight() + ImGui::GetStyle().FramePadding.y * 2.f;
    const float btnPosX = ImGui::GetContentRegionAvail().x - btnWidth;
    
    ImGui::SameLine( btnPosX );
    if ( UI::button( "Remove", selectedFileName_ != defaultName, {btnWidth, btnHeight}) )
    {
        std::filesystem::remove( getFolder_() / ( selectedFileName_ + ".mrmesh" ) );
        selectedFileName_ = defaultName;
        result = true;
        toolMesh_ = defaultToolMesh_;
    }

    if ( openSelectMeshPopup )
        ImGui::OpenPopup( "SelectMesh" );
    drawSelectMeshPopup_();
    return result;
}

std::filesystem::path GcodeToolsLibrary::getFolder_()
{
    return getUserConfigDir() / libraryName_;
}

void GcodeToolsLibrary::updateFilesList_()
{
    filesList_.clear();
    for ( const auto& entry : std::filesystem::directory_iterator( getFolder_() ) )
    {
        const auto filename = entry.path().filename();
        if ( utf8string( filename.extension() ) == ".mrmesh" )
            filesList_.push_back( utf8string( filename.stem() ) );
    }
}

void GcodeToolsLibrary::addNewToolFromFile_()
{
    auto path = openFileDialog( { .filters = MeshLoad::getFilters() } );
    if ( path.empty() )
        return;

    auto loadRes = MeshLoad::fromAnySupportedFormat( path );
    if ( !loadRes )
        return;

    toolMesh_ = std::make_shared<ObjectMesh>();
    toolMesh_->setName( utf8string( path.filename().stem() ) );
    toolMesh_->setMesh( std::make_shared<Mesh>( *loadRes ) );
    MeshSave::toMrmesh( *loadRes, getFolder_() / ( toolMesh_->name() + ".mrmesh" ) );
    selectedFileName_ = toolMesh_->name();
}

void GcodeToolsLibrary::addNewToolFromMesh_( const std::shared_ptr<ObjectMesh>& objMesh )
{
    toolMesh_ = std::dynamic_pointer_cast< ObjectMesh >( objMesh->clone() );
    MeshSave::toMrmesh( *toolMesh_->mesh(), getFolder_() / ( toolMesh_->name() + ".mrmesh" ) );
    updateFilesList_();
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
    auto path = getFolder_() / ( filename + ".mrmesh" );
    if ( !std::filesystem::exists( path ) )
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
