#pragma once
#ifndef MESHLIB_NO_VOXELS
#include "MRMesh/MRMeshFwd.h"
#include "MRCommonPlugins/exports.h"
#include "MRViewer/MRStatePlugin.h"
#include "MRViewer/MRSceneStateCheck.h"
#include "MRViewer/MRStatePluginUpdate.h"
#include "MRVoxels/MRObjectVoxels.h"

namespace MR
{

class BinaryOperations : public StatePlugin, public SceneStateExactCheck<2, ObjectVoxels>,
    public PluginCloseOnSelectedObjectRemove
{
public:
    BinaryOperations();

    virtual void drawDialog(float menuScaling, ImGuiContext*) override;

private:

    virtual bool onEnable_() override;
    virtual bool onDisable_() override;

    enum class Operation : int
    {
        Union,
        Intersection,
        Difference,
        Max,
        Min,
        Sum,
        Mul,
        Div,
        Count
    };
    void doOperation_( Operation op, bool inPreview );

    void onTransformChange();

    std::shared_ptr<ObjectVoxels> obj1_;
    std::shared_ptr<ObjectVoxels> obj2_;
    boost::signals2::scoped_connection conn1_;
    boost::signals2::scoped_connection conn2_;

    std::vector<std::string> enabledOps_;
    std::vector<std::string> enabledOpsTooltips_;

    std::shared_ptr<ObjectVoxels> previewRes_;
    bool previewMode_ = false;
    Operation operation_ = Operation::Union;
};

}
#endif
