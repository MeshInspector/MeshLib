#pragma once
#include "MRViewer/MRRibbonMenuItem.h"

namespace MR
{

class RibbonSceneSortByName : public RibbonMenuItem
{
public:
    RibbonSceneSortByName();

    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& objs ) const override;

    // returns true if state of item changed
    virtual bool action() override;
private:
    void sortObjectsRecursive_( std::shared_ptr<Object> object );
};

class RibbonSceneSelectAll : public RibbonMenuItem
{
public:
    RibbonSceneSelectAll();

    // returns true if state of item changed
    virtual bool action() override;
};

class RibbonSceneUnselectAll : public RibbonMenuItem, public SceneStateAtLeastCheck<1, Object>
{
public:
    RibbonSceneUnselectAll();

    // returns true if state of item changed
    virtual bool action() override;
};

class RibbonSceneShowOnlyPrev : public RibbonMenuItem
{
public:
    RibbonSceneShowOnlyPrev();

    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& objs ) const override;

    // returns true if state of item changed
    virtual bool action() override;
};

class RibbonSceneShowOnlyNext : public RibbonMenuItem
{
public:
    RibbonSceneShowOnlyNext();

    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& objs ) const override;

    // returns true if state of item changed
    virtual bool action() override;
};

class RibbonSceneRename : public RibbonMenuItem, public SceneStateExactCheck<1, Object>
{
public:
    RibbonSceneRename();

    // returns true if state of item changed
    virtual bool action() override;
};

class RibbonSceneRemoveSelected : public RibbonMenuItem, public SceneStateAtLeastCheck<1, Object>
{
public:
    RibbonSceneRemoveSelected();

    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& objs ) const override;

    // returns true if state of item changed
    virtual bool action() override;
};


}
