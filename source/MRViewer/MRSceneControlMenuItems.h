#pragma once
#include "MRRibbonMenuItem.h"
#include "MRHistoryStore.h"

namespace MR
{

class UndoMenuItem : public RibbonMenuItem
{
public:
    UndoMenuItem();
    virtual bool action() override;
    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const override;
    virtual std::string getDynamicTooltip() const override;
    
private:
    void updateUndoListCache_( const HistoryStore& store, HistoryStore::ChangeType type );

    boost::signals2::scoped_connection historyStoreConnection_;
};

class RedoMenuItem : public RibbonMenuItem
{
public:
    RedoMenuItem();
    virtual bool action() override;
    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const override;
    virtual std::string getDynamicTooltip() const override;

private:
    void updateRedoListCache_( const HistoryStore& store, HistoryStore::ChangeType type );

    boost::signals2::scoped_connection historyStoreConnection_;
};

}