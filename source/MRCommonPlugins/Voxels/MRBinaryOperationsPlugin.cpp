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

const std::string operationNames[] = {
    "Union",
    "Intersection",
    "Difference",
    "Max",
    "Min",
    "Sum",
    "Multiply",
    "Divide",
    "Replace"
};

const std::string operationTooltips[] = {
    "Union A + B",
    "Intersection A * B",
    "Difference A - B",
    "Compute max(a, b) per voxel",
    "Compute min(a, b) per voxel",
    "Compute a + b per voxel",
    "Compute a * b per voxel",
    "Compute a / b per voxel",
    "Copy the active voxels of B into A"
};

void BinaryOperations::drawDialog(float menuScaling, ImGuiContext*)
{
    auto menuWidth = 200 * menuScaling;
    if (!ImGuiBeginWindow_( { .width = menuWidth, .menuScaling = menuScaling }))
        return;

    ImGui::Text("Object A: %s", obj1_->name().c_str());
    ImGui::Text("Object B: %s", obj2_->name().c_str());

    if (UI::button("Swap", { -1, 0 }))
        std::swap(obj1_, obj2_);

    UI::separator(menuScaling, "Operations");

    for (int i = 0; i < int(Operation::Count); ++i)
    {
        if (UI::button(operationNames[i].c_str(), { -1, 0 }))
        {
            doOperation_(Operation(i));
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", operationTooltips[i].c_str());
    }
    ImGui::EndCustomStatePlugin();
}

bool BinaryOperations::onEnable_()
{
    auto objs = getAllObjectsInTree<ObjectVoxels>(&SceneRoot::get(), ObjectSelectivityType::Selected);
    obj1_ = objs[0];
    obj2_ = objs[1];
    return true;
}

bool BinaryOperations::onDisable_()
{
    obj1_.reset();
    obj2_.reset();
    return true;
}

void BinaryOperations::doOperation_(Operation op)
{
    ProgressBar::orderWithMainThreadPostProcessing(operationNames[int(op)].c_str(), [&, this, op]()->std::function<void()>
    {
        std::function<void()> cancelRes = []
        {
            showError(stringOperationCanceled());
        };

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
                if (!ProgressBar::setProgress(0.25f))
                    return cancelRes;
                openvdb::FloatGrid::Ptr copy2 = grid2.deepCopy();
                if (!ProgressBar::setProgress(0.5f))
                    return cancelRes;
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
                case MR::BinaryOperations::Operation::Replace:
                    openvdb::tools::compReplace( *resGrid, *copy2 );
                    resIso = iso2;
                    break;
                default:
                    assert( false );
                }
            }
        }
        if (!ProgressBar::setProgress(0.75f))
            return cancelRes;

        std::shared_ptr<ObjectVoxels> newObj = std::make_shared<ObjectVoxels>();
        newObj->setName(operationNames[int(op)]);
        newObj->construct( MakeFloatGrid( std::move( resGrid ) ), obj1_->vdbVolume().voxelSize );
        newObj->setXf( postXf );
        if ( !newObj->setIsoValue( resIso, subprogress(ProgressBar::setProgress, 0.75f, 1.f)) )
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
    });
}

MR_REGISTER_RIBBON_ITEM(BinaryOperations)

} //namespace MR

#endif
