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

    enum class Operation
    {
        Union,
        Intersection,
        Difference,
        Max,
        Min,
        Sum,
        Mul,
        Div,
        Replace,
        Count
    };
    void doOperation_(Operation op);

    std::shared_ptr<ObjectVoxels> obj1_;
    std::shared_ptr<ObjectVoxels> obj2_;
};

}
#endif
