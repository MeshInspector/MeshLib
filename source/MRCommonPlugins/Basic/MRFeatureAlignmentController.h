#pragma once
#include "MRFeatureAlignmentScene.h"
#include <boost/signals2/connection.hpp>

namespace MR
{
class HistoryStore;
} // namespace MR

namespace MR::FeatureAlignmentTool
{
class Controller
{
public:
    struct ViewState
    {
        Catalog entries;
        // Reference, Floating, in that order.
        std::array<Catalog, 2> primaryFeatures, locatorFeatures, clockingFeatures;
        bool canApply{}, canSwapRoles{};
        std::string setupError;
    };
    explicit Controller( ObjectPtr root );
    const ObjectPtr &root() const { return root_; }
    const Selection &selection() const { return selection_; }
    const Settings &settings() const { return settings_; }
    const std::optional<Solution> &result() const { return result_; }
    const ViewState &viewState();
    void refresh();
    void configure( Selection selection, Settings settings );
    Expected<void> swapRoles();
    Expected<void> apply();

private:
    void watchSelected_();
    void watchHistory_();
    void selectedChanged_();
    Expected<AffineXf3f> evaluate_( const Catalog &entries );
    bool canApply_() const;
    Expected<void> validateSwap_( const Catalog &entries ) const;
    ObjectPtr root_;
    Selection selection_;
    Settings settings_;
    ViewState viewState_;
    bool viewDirty_{ true };
    std::optional<Solution> result_;
    struct WatchedObject
    {
        ObjectPtr object;
        const Object *parent{};
        size_t childCount{};
        bool operator==( const WatchedObject & ) const = default;
    };
    std::vector<WatchedObject> watchedObjects_;
    std::vector<boost::signals2::scoped_connection> connections_;
    HistoryStore *history_{};
    boost::signals2::scoped_connection historyConnection_;
};
} // namespace MR::FeatureAlignmentTool
