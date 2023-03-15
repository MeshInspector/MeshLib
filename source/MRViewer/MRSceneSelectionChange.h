#pragma once
#include "exports.h"
#include "MRMesh/MRObject.h"
#include <memory>
#include <vector>

namespace MR
{

// Interface for processing scene state change in RibbonItems
class ISceneSelectionChange
{
public:
    virtual ~ISceneSelectionChange() = default;
    // do something on selection change,
    // args - newly selected objects
    virtual void updateSelection( const std::vector<std::shared_ptr<const Object>>& ) {}
};

// close on state change
class MRVIEWER_CLASS SceneSelectionChangeClose : virtual public ISceneSelectionChange
{
public:
    MRVIEWER_API virtual void updateSelection( const std::vector<std::shared_ptr<const Object>>& ) override;
};

// restart on state change
class MRVIEWER_CLASS SceneSelectionChangeRestart : virtual public ISceneSelectionChange
{
public:
    MRVIEWER_API virtual void updateSelection( const std::vector<std::shared_ptr<const Object>>& ) override;
};

}
