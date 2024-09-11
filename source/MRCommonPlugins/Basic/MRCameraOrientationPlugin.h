#pragma once

#include "MRMesh/MRMeshFwd.h"
#include "MRViewer/MRStatePlugin.h"
#include "MRMesh/MRVector3.h"

namespace MR
{

class CameraOrientation : public MR::StatePlugin
{
public:
    CameraOrientation();

    virtual void drawDialog( float menuScaling, ImGuiContext* ) override;

    virtual bool blocking() const override { return false; }
private:

    Vector3f position_;
    Vector3f direction_{ 1.f, 0.f, 0.f };
    Vector3f upDir_{ 0.f, 1.f, 0.f };
    bool isAutofit_{ true };

    virtual bool onEnable_() override;

    void drawCameraPresets_( float scaling );

    inline void autofit_();
};

}
