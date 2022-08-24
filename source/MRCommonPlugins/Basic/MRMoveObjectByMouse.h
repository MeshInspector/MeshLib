#pragma once
#include "MRViewer/MRStatePlugin.h"
#include "MRMesh/MRPlane3.h"
#include "imgui.h"

namespace MR
{

class Object;

class MoveObjectByMouse : public StateListenerPlugin<MouseDownListener, MouseMoveListener, MouseUpListener>
{
public:
    MoveObjectByMouse();

    virtual void drawDialog( float menuScaling, ImGuiContext* ) override;

    virtual bool blocking() const override { return false; };

private:
    virtual bool onMouseDown_( MouseButton button, int modifier ) override;
    virtual bool onMouseMove_( int x, int y ) override;
    virtual bool onMouseUp_( MouseButton btn, int modifiers ) override;

    void setVisualizeVectors_( std::vector<Vector3f> worldPoints );

    std::shared_ptr<VisualObject> obj_;

    Vector3f worldStartPoint_;
    Vector3f worldBboxCenter_;
    Vector3f bboxCenter_;
    AffineXf3f objXf_;
    float viewportStartPointZ_;
    Plane3f rotationPlane_;

    std::vector<ImVec2> visualizeVectors_;
    float angle_ = 0.f;
    float shift_ = 0.f;

    enum class TransformMode
    {
        Translation,
        Rotation,
        None
    } transformMode_ = TransformMode::None;
};

}
