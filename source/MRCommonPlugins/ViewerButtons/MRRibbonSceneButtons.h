#pragma once

#include "MRViewer/MRRibbonMenuItem.h"
#include "MRViewer/MRSceneStateCheck.h"
#include "MRMesh/MRObject.h"

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

    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& objs ) const override;

    // returns true if state of item changed
    virtual bool action() override;
};

class RibbonSceneUnselectAll : public RibbonMenuItem, public SceneStateAtLeastCheck<1, Object, NoModelCheck>
{
public:
    RibbonSceneUnselectAll();

    // returns true if state of item changed
    virtual bool action() override;
};

class RibbonSceneShowAll : public RibbonMenuItem
{
public:
    RibbonSceneShowAll();

    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& objs ) const override;

    // returns true if state of item changed
    virtual bool action() override;
};

class RibbonSceneHideAll : public RibbonMenuItem
{
public:
    RibbonSceneHideAll();

    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& objs ) const override;

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

class RibbonSceneRename : public RibbonMenuItem, public SceneStateExactCheck<1, Object, NoModelCheck>
{
public:
    RibbonSceneRename();

    // returns true if state of item changed
    virtual bool action() override;
};

class RibbonSceneRemoveSelected : public RibbonMenuItem, public SceneStateAtLeastCheck<1, Object, NoModelCheck>
{
public:
    RibbonSceneRemoveSelected();

    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& objs ) const override;

    // returns true if state of item changed
    virtual bool action() override;
};


}
