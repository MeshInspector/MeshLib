#ifndef MESHLIB_NO_VOXELS
#include "MRBinaryOperationsPlugin.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRViewer/MRUIStyle.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRVoxels/MRVDBFloatGrid.h"
#include "MRVoxels/MRFloatGrid.h"
#include "MRVoxels/MRObjectVoxels.h"
#include "MRViewer/MRAppendHistory.h"
#include "MRViewer/MRProgressBar.h"
#include "MRMesh/MRMatrix4.h"
#include "MRVoxels/MRVoxelsApplyTransform.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRChangeObjectFields.h"
#include "MRMesh/MRChangeSceneAction.h"
#include "MRMesh/MRObjectsAccess.h"

namespace MR
{

BinaryOperations::BinaryOperations() :
    StatePlugin( "Binary Operations" )
{
}

const std::vector<std::string> operationNames = {
    "Union",
    "Intersection",
    "Difference",
    "Max",
    "Min",
    "Sum",
    "Multiply",
    "Divide"
};

const std::vector<std::string> operationTooltips = {
    "Union A + B",
    "Intersection A * B",
    "Difference A - B",
    "Compute max(a, b) per voxel",
    "Compute min(a, b) per voxel",
    "Compute a + b per voxel",
    "Compute a * b per voxel",
    "Compute a / b per voxel"
};

void BinaryOperations::drawDialog(float menuScaling, ImGuiContext*)
{
    auto menuWidth = 300 * menuScaling;
    if (!ImGuiBeginWindow_( { .width = menuWidth, .menuScaling = menuScaling }))
        return;

    const auto& style = ImGui::GetStyle();
    const float textWidth = menuWidth - 2 * style.WindowPadding.x - style.ItemSpacing.x - ImGui::CalcTextSize( "Reference object" ).x;
    UI::inputTextCenteredReadOnly( "Object A", obj1_->name(), textWidth, ImVec4{ 1.0f, 0.4f, 0.4f, 1.0f } );
    UI::inputTextCenteredReadOnly( "Object B", obj2_->name(), textWidth, ImVec4{ 0.4f, 0.4f, 1.0f, 1.0f } );

    if ( UI::button( "Swap", {-1, 0} ) )
    {
        std::swap( obj1_, obj2_ );
        std::swap( conn1_, conn2_ );
        if ( previewMode_ )
            doOperation_( operation_, true );
    }

    UI::separator(menuScaling, "Operations");

    if ( UI::checkbox( "Enable preview", &previewMode_ ) )
    {
        if ( previewMode_ )
        {
            obj1_->setVisualizeProperty( true, MeshVisualizePropertyType::OnlyOddFragments, ViewportMask::all() );
            obj2_->setVisualizeProperty( true, MeshVisualizePropertyType::OnlyOddFragments, ViewportMask::all() );
            if ( !previewRes_ )
            {
                previewRes_ = std::make_shared<ObjectVoxels>();
                previewRes_->setAncillary( true );
                SceneRoot::get().addChild( previewRes_ );
            }
            previewRes_->setName( operationNames[(int)operation_] );
            doOperation_( operation_, true );
        }
        else
        {
            obj1_->setVisualizeProperty( false, MeshVisualizePropertyType::OnlyOddFragments, ViewportMask::all() );
            obj2_->setVisualizeProperty( false, MeshVisualizePropertyType::OnlyOddFragments, ViewportMask::all() );
            if ( previewRes_ )
                SceneRoot::get().removeChild( previewRes_ );
            previewRes_.reset();
        }
    }

    if ( UI::combo( "Operation", (int*)&operation_, operationNames, true, operationTooltips ) )
    {
        if ( previewMode_ )
            doOperation_( operation_, true );
    }

    if ( UI::button( "Apply", { -1, 0 } ) )
    {
        doOperation_( operation_, false );
    }

    ImGui::EndCustomStatePlugin();
}

bool BinaryOperations::onEnable_()
{
    auto objs = getAllObjectsInTree<ObjectVoxels>(&SceneRoot::get(), ObjectSelectivityType::Selected);
    obj1_ = objs[0];
    obj2_ = objs[1];
    conn1_ = obj1_->worldXfChangedSignal.connect( [this] { return onTransformChange(); } );
    conn2_ = obj2_->worldXfChangedSignal.connect( [this] { return onTransformChange(); } );
    return true;
}

bool BinaryOperations::onDisable_()
{
    conn1_.disconnect();
    conn1_.disconnect();
    obj1_->setVisualizeProperty( false, MeshVisualizePropertyType::OnlyOddFragments, ViewportMask::all() );
    obj2_->setVisualizeProperty( false, MeshVisualizePropertyType::OnlyOddFragments, ViewportMask::all() );
    obj1_.reset();
    obj2_.reset();
    if ( previewRes_ )
        SceneRoot::get().removeChild( previewRes_ );
    previewRes_.reset();
    previewMode_ = false;
    return true;
}

void BinaryOperations::onTransformChange()
{
    if ( previewMode_ )
    {
        doOperation_( operation_, true );
    }
}

void BinaryOperations::doOperation_( Operation op, bool inPreview )
{
    struct Res
    {
        FloatGrid grid;
        AffineXf3f xf;
        float iso = 0;
    };

    auto func = [&, this, op] ( auto reportProgress ) -> std::optional<Res>
    {
        VdbVolume vol1 = obj1_->vdbVolume();
        VdbVolume vol2 = obj2_->vdbVolume();
        openvdb::FloatGrid::Ptr resGrid;
        const float iso1 = obj1_->getIsoValue();
        const float iso2 = obj2_->getIsoValue();
        float resIso = iso1;

        // select the smallest object. In case of a not identity transformations, it will be transformed to the space of the largest object
        auto xf = obj1_->xf().inverse() * obj2_->xf();
        auto postXf = xf;
        if ( xf != AffineXf3f() )
        {
            if ( vol1.data->activeVoxelCount() < vol2.data->activeVoxelCount() )
            {
                vol1 = transformVdbVolume( vol1, xf.inverse() ).volume;
                postXf = obj2_->xf();
            }
            else
            {
                vol2 = transformVdbVolume( vol2, xf ).volume;
                postXf = obj1_->xf();
            }
        }

        auto& grid1 = *vol1.data;
        auto& grid2 = *vol2.data;

        switch (op)
        {
        case MR::BinaryOperations::Operation::Union:
            resGrid = openvdb::tools::csgUnionCopy(ovdb(grid1), ovdb(grid2));
            break;
        case MR::BinaryOperations::Operation::Intersection:
            resGrid = openvdb::tools::csgIntersectionCopy(ovdb(grid1), ovdb(grid2));
            break;
        case MR::BinaryOperations::Operation::Difference:
            resGrid = openvdb::tools::csgDifferenceCopy(ovdb(grid1), ovdb(grid2));
            break;
        default:
            {
                resGrid = grid1.deepCopy();
                if (!reportProgress(0.25f))
                    return {};
                openvdb::FloatGrid::Ptr copy2 = grid2.deepCopy();
                if (!reportProgress(0.5f))
                    return {};
                switch (op)
                {
                case MR::BinaryOperations::Operation::Max:
                    openvdb::tools::compMax( *resGrid, *copy2 );
                    resIso = std::max( iso1, iso2 );
                    break;
                case MR::BinaryOperations::Operation::Min:
                    openvdb::tools::compMin( *resGrid, *copy2 );
                    resIso = std::min( iso1, iso2 );
                    break;
                case MR::BinaryOperations::Operation::Sum:
                    openvdb::tools::compSum( *resGrid, *copy2 );
                    resIso = iso1 + iso2;
                    break;
                case MR::BinaryOperations::Operation::Mul:
                    openvdb::tools::compMul( *resGrid, *copy2 );
                    resIso = iso1 * iso2;
                    break;
                case MR::BinaryOperations::Operation::Div:
                    openvdb::tools::compDiv( *resGrid, *copy2 );
                    if ( iso2 != 0 )
                        resIso = iso1 / iso2;
                    break;
                default:
                    assert( false );
                }
            }
        }
        if (!reportProgress(0.75f))
            return {};

        return Res{
            .grid = MakeFloatGrid( std::move( resGrid ) ),
            .xf = postXf,
            .iso = resIso
        };
    };

    if ( inPreview )
    {
        if ( auto res = func( [] ( float ) { return true; } ) )
        {
            previewRes_->setName( operationNames[(int)op] );
            previewRes_->setXf( res->xf );
            previewRes_->construct( res->grid, obj1_->vdbVolume().voxelSize );
            previewRes_->updateIsoSurface( *previewRes_->recalculateIsoSurface( res->iso ) );
        }
    }
    else
        ProgressBar::orderWithMainThreadPostProcessing(operationNames[int(op)].c_str(), [this, op, func] () -> std::function<void()> {
            std::function<void()> cancelRes = []
            {
                showError(stringOperationCanceled());
            };

            auto res = func( ProgressBar::setProgress );
            if ( !res )
                return cancelRes;
            std::shared_ptr<ObjectVoxels> newObj = std::make_shared<ObjectVoxels>();
            newObj->setName(operationNames[int(op)]);
            newObj->construct( res->grid, obj1_->vdbVolume().voxelSize );
            newObj->setXf( res->xf );
            if ( !newObj->setIsoValue( res->iso, subprogress(ProgressBar::setProgress, 0.75f, 1.f)) )
                return cancelRes;
            return [this, newObj]()
            {
                SCOPED_HISTORY( newObj->name() );

                AppendHistory<ChangeObjectVisibilityAction>( "invis1", obj1_ );
                obj1_->setVisible( false );
                AppendHistory<ChangeObjectSelectedAction>( "unselect1", obj1_ );
                obj1_->select( false );

                AppendHistory<ChangeObjectVisibilityAction>( "invis2", obj2_ );
                obj2_->setVisible( false );
                AppendHistory<ChangeObjectSelectedAction>( "unselect2", obj2_ );
                obj2_->select( false );

                AppendHistory<ChangeSceneAction>( "add obj", newObj, ChangeSceneAction::Type::AddObject );
                newObj->select( true );
                SceneRoot::get().addChild( newObj );
                dialogIsOpen_ = false;
            };
        } );
}

MR_REGISTER_RIBBON_ITEM(BinaryOperations)

} //namespace MR

#endif
