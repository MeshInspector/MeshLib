#pragma once
#include "MRFeatureAlignmentScene.h"
#include <boost/signals2/connection.hpp>
#include <unordered_set>

namespace MR
{
class HistoryStore;
class CombinedHistoryAction;
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
        bool canPreview{}, canApply{}, stalePreview{}, canSwapRoles{};
        std::string setupError;
    };
    explicit Controller( ObjectPtr root );
    ~Controller();
    const ObjectPtr &root() const { return root_; }
    const Selection &selection() const { return selection_; }
    const Settings &settings() const { return settings_; }
    const std::optional<Solution> &result() const { return result_; }
    const ViewState &viewState();
    void refresh();
    Expected<void> configure( Selection selection, Settings settings );
    Expected<void> swapRoles();
    Expected<void> invalidate( bool abandonChangedPreview = false );
    Expected<Solution> preview();
    Expected<void> apply();
    bool canApply() const;
    bool stalePreview() const;
    bool hasPreview() const { return bool( preview_ ); }

private:
    void watchSelected_();
    void watchHistory_();
    void selectedChanged_();
    Expected<void> cancelPreview_( bool abandonChangedPreview, CombinedHistoryAction *pendingGroup = nullptr );
    Expected<void> validateSwap_( const Catalog &entries ) const;
    ObjectPtr root_;
    Selection selection_;
    Settings settings_;
    ViewState viewState_;
    bool viewDirty_{ true }, previewChanged_{}, changingPose_{};
    struct Preview
    {
        ObjectPtr moving;
        const Object *parent{};
        AffineXf3f before, target;
        std::unordered_set<ObjectPtr> members;
    };
    std::optional<Preview> preview_;
    std::optional<Solution> result_;
    std::vector<ObjectPtr> watchedObjects_;
    std::vector<boost::signals2::scoped_connection> connections_;
    HistoryStore *history_{};
    boost::signals2::scoped_connection historyConnection_;
};
} // namespace MR::FeatureAlignmentTool
