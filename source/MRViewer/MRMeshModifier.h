#pragma once
#include "exports.h"
#include "MRStatePlugin.h"
#include <MRMesh/MRVisualObject.h>

namespace MR
{
class RenderObject;
struct Mesh;
class MRVIEWER_CLASS MeshModifier : public RibbonMenuItem
{
public:
    MRVIEWER_API MeshModifier( std::string name, StatePluginTabs tab = StatePluginTabs::Mesh );
    virtual ~MeshModifier() = default;

    MRVIEWER_API virtual bool action() override;

    // RenderObject here for auto update
    MRVIEWER_API bool modify( const std::vector<std::shared_ptr<VisualObject>>& selectedObjects );

    MRVIEWER_API StatePluginTabs getTab() const;

    // check if search mask satisfies for this modifier
    MRVIEWER_API bool checkStringMask( const std::string& mask ) const;

private:
    virtual bool modify_( const std::vector<std::shared_ptr<VisualObject>>& selectedObjects ) = 0;
    StatePluginTabs tab_;
};
}
