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
#include "ImGuiHelpers.h"
#include "MRMesh/MREndMill.h"
#include "MRMesh/MRSerializer.h"
#include "MRPch/MRJson.h"

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
                result = loadFromFile_( filesList_[i] );
            }
        }

        selected = false;
        if ( ImGui::Selectable( "<Create Tool>", &selected ) )
        {
            createToolDialogIsOpen_ = true;
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
        removeSelectedTool_();
        result = true;
    }

    return result;
}

bool GcodeToolsLibrary::drawCreateToolDialog( float menuScaling )
{
    if ( !createToolDialogIsOpen_ )
        return false;

    const auto menuWidth = 220.f * menuScaling;
    if ( !ImGui::BeginCustomStatePlugin( "Create Tool", &createToolDialogIsOpen_, {
        .width = menuWidth,
        .menuScaling = menuScaling,
    } ) )
        return false;

    bool result = false;

    const auto itemWidth = 160.f * menuScaling;

    UI::inputTextCentered( "Name", createToolName_, itemWidth );

    static const std::vector<std::string> cToolTypeNames {
        "Flat End Mill",
        "Ball End Mill",
        "Bull Nose End Mill",
        "Chamfer End Mill",
    };
    assert( cToolTypeNames.size() == (int)EndMillCutter::Type::Count );
    ImGui::SetNextItemWidth( itemWidth );
    UI::combo( "Type", &createToolType_, cToolTypeNames );

    UI::separator( menuScaling, "Specifications" );

    ImGui::PushItemWidth( 115.f * menuScaling );
    UI::drag<LengthUnit>( "Length", createToolLength_, 1e-3f, 1e-3f, 1e+3f );
    UI::drag<LengthUnit>( "Diameter", createToolDiameter_, 1e-3f, 1e-3f, 1e+3f );
    if ( createToolType_ == (int)EndMillCutter::Type::Ball )
    {
        auto radius = createToolDiameter_ / 2.f;
        if ( UI::drag<LengthUnit>( "Cutter Radius", radius, 1e-3f, 1e-3f, 1e+3f ) )
            createToolDiameter_ = radius * 2.f;
    }
    if ( createToolType_ == (int)EndMillCutter::Type::BullNose )
    {
        UI::drag<LengthUnit>( "Cutter Radius", createToolCornerRadius_, 1e-3f, 0.f, createToolDiameter_ / 2.f );
    }
    if ( createToolType_ == (int)EndMillCutter::Type::Chamfer )
    {
        UI::drag<AngleUnit>( "Cutting Angle", createToolCuttingAngle_, 1.f, 0.f, 180.f, { .sourceUnit = AngleUnit::degrees } );
        UI::drag<LengthUnit>( "End Diameter", createToolEndDiameter_, 1e-3f, 0.f, createToolDiameter_ );
    }
    ImGui::PopItemWidth();

    // TODO: visualize tool

    const auto isValid = !createToolName_.empty() && createToolLength_ > 0.f && createToolDiameter_ > 0.f;
    if ( UI::button( "Create", isValid, { -1.f, 0.f } ) )
    {
        addNewTool_( createToolName_, {
            .length = createToolLength_,
            .diameter = createToolDiameter_,
            .cutter = EndMillCutter {
                .type = (EndMillCutter::Type)createToolType_,
                .cornerRadius = createToolType_ == (int)EndMillCutter::Type::BullNose ? createToolCornerRadius_ : 0.f,
                .cuttingAngle = createToolType_ == (int)EndMillCutter::Type::Chamfer ? createToolCuttingAngle_ : 0.f,
                .endDiameter = createToolType_ == (int)EndMillCutter::Type::Chamfer ? createToolEndDiameter_ : 0.f,
            },
        } );
        createToolDialogIsOpen_ = false;
        result = true;
    }

    ImGui::EndCustomStatePlugin();

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
        const auto extension = utf8string( filename.extension() );
        if ( extension == ".mrmesh" || extension == ".json" )
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
    (void)MeshSave::toMrmesh( *loadRes, folderPath / ( toolMesh_->name() + ".mrmesh" ) ); //TODO: process potential error
    endMillTool_.reset();
    selectedFileName_ = toolMesh_->name();
}

void GcodeToolsLibrary::addNewToolFromMesh_( const std::shared_ptr<ObjectMesh>& objMesh )
{
    const auto folderPath = getFolder_();
    if ( folderPath.empty() )
        return;

    toolMesh_ = std::dynamic_pointer_cast< ObjectMesh >( objMesh->clone() );
    (void)MeshSave::toMrmesh( *toolMesh_->mesh(), folderPath / ( toolMesh_->name() + ".mrmesh" ) ); //TODO: process potential error
    endMillTool_.reset();
    selectedFileName_ = toolMesh_->name();
}

void GcodeToolsLibrary::addNewTool_( const std::string& name, const EndMillTool& tool )
{
    const auto folderPath = getFolder_();
    if ( folderPath.empty() )
        return;

    Json::Value root;
    serializeToJson( tool, root );
    if ( !serializeJsonValue( root, folderPath / ( name + ".json" ) ) )
        return;

    toolMesh_ = std::make_shared<ObjectMesh>();
    toolMesh_->setName( name );
    toolMesh_->setMesh( std::make_shared<Mesh>( tool.toMesh() ) );

    endMillTool_ = std::make_shared<EndMillTool>( tool );

    selectedFileName_ = name;
}

void GcodeToolsLibrary::removeSelectedTool_()
{
    const auto folderPath = getFolder_();
    if ( folderPath.empty() )
        return;

    std::error_code ec;
    std::filesystem::remove( folderPath / ( selectedFileName_ + ".mrmesh" ), ec );
    std::filesystem::remove( folderPath / ( selectedFileName_ + ".json" ), ec );

    toolMesh_ = defaultToolMesh_;
    endMillTool_.reset();
    selectedFileName_ = defaultName;
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

bool GcodeToolsLibrary::loadFromFile_( const std::string& filename )
{
    const auto folderPath = getFolder_();
    if ( folderPath.empty() )
        return false;

    std::error_code ec;

    const auto meshPath = folderPath / ( filename + ".mrmesh" );
    if ( std::filesystem::exists( meshPath, ec ) )
    {
        auto loadRes = MeshLoad::fromMrmesh( meshPath );
        if ( !loadRes )
            return false;

        toolMesh_ = std::make_shared<ObjectMesh>();
        toolMesh_->setName( filename );
        toolMesh_->setMesh( std::make_shared<Mesh>( *loadRes ) );

        endMillTool_.reset();

        selectedFileName_ = filename;
        return true;
    }

    const auto jsonPath = folderPath / ( filename + ".json" );
    if ( std::filesystem::exists( jsonPath, ec ) )
    {
        auto loadRes = deserializeJsonValue( jsonPath );
        if ( !loadRes )
            return false;

        EndMillTool tool;
        if ( auto res = deserializeFromJson( *loadRes, tool ); !res )
            return false;

        toolMesh_ = std::make_shared<ObjectMesh>();
        toolMesh_->setName( filename );
        toolMesh_->setMesh( std::make_shared<Mesh>( tool.toMesh() ) );

        endMillTool_ = std::make_shared<EndMillTool>( tool );

        selectedFileName_ = filename;
        return true;
    }

    return false;
}

}
