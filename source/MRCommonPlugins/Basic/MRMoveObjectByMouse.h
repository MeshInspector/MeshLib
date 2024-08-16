#pragma once
#include "MRViewer/MRStatePlugin.h"
#include "MRViewer/MRMoveObjectByMouseImpl.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRCommonPlugins/exports.h"
#include "imgui.h"

namespace MR
{

class Object;

class MoveObjectByMouse : public StateListenerPlugin<DragStartListener, DragListener, DragEndListener>
{
public:
    MoveObjectByMouse();
    ~MoveObjectByMouse();

    MRCOMMONPLUGINS_API static MoveObjectByMouse* instance();

    virtual bool onDisable_() override;
    virtual void drawDialog( float menuScaling, ImGuiContext* ) override;

    virtual bool blocking() const override { return false; };

private:
    virtual bool onDragStart_( MouseButton btn, int modifiers ) override;
    virtual bool onDrag_( int x, int y ) override;
    virtual bool onDragEnd_( MouseButton btn, int modifiers ) override;

    // Same as basic implementation but allows to move selected objects together by holding Shift
    class MoveObjectByMouseWithSelected : public MoveObjectByMouseImpl
    {
    protected:
        TransformMode pick_( MouseButton button, int modifiers,
            std::vector<std::shared_ptr<Object>>& objects, Vector3f& centerPoint, Vector3f& startPoint ) override;
    } moveByMouse_;
};

}
