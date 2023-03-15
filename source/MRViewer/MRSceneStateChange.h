#pragma once
#include "MRSceneStateCheck.h"

namespace MR
{

// Interface for processing scene state change in RibbonItems
class ISceneStateChange
{
public:
    virtual ~ISceneStateChange() = default;
    // do something on selection change
    virtual void updateSelection( const std::vector<std::shared_ptr<const Object>>& ) {}
};

// close on state change
class MRVIEWER_CLASS SceneStateChangeClose : virtual public ISceneStateChange
{
public:
    MRVIEWER_API virtual void updateSelection( const std::vector<std::shared_ptr<const Object>>& ) override;
};

// restart on state change
class MRVIEWER_CLASS SceneStateChangeRestart : virtual public ISceneStateChange
{
public:
    MRVIEWER_API virtual void updateSelection( const std::vector<std::shared_ptr<const Object>>& ) override;
};

}
