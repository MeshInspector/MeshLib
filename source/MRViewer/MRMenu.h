#pragma once
#include "ImGuiMenu.h"

#include "imgui.h"
#include "exports.h"


namespace MR
{


class [[deprecated]] Menu : public MR::ImGuiMenu
{
public:
    MRVIEWER_API virtual void init( MR::Viewer *_viewer ) override;
};

}
