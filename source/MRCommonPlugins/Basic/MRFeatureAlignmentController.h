#pragma once
#include "MRFeatureAlignmentScene.h"
#include <boost/signals2/connection.hpp>

namespace MR::FeatureAlignmentTool
{
class TransformHistory
{
public:
    virtual ~TransformHistory() = default;
    virtual Expected<void> commit( const ObjectPtr &object, const AffineXf3f &local, const std::string &label ) = 0;
    Expected<void> commitPreview( const ObjectPtr &object, const AffineXf3f &before, const AffineXf3f &target,
                                  const std::string &label );
};
class ViewerTransformHistory : public TransformHistory
{
public:
    Expected<void> commit( const ObjectPtr &, const AffineXf3f &, const std::string & ) override;
};
class Controller
{
public:
    Controller( ObjectPtr root, TransformHistory &history );
    ~Controller();
    const ObjectPtr &root() const
    {
        return root_;
    }
    const Selection &selection() const
    {
        return selection_;
    }
    const Settings &settings() const
    {
        return settings_;
    }
    Expected<void> configure( Selection selection, Settings settings );
    Expected<void> invalidate( bool abandonChangedPreview = false );
    Expected<Solution> preview( bool show = false );
    Expected<void> apply();
    Expected<void> restore();
    bool canApply() const;
    bool canRestore() const;
    bool stalePreview() const;
    bool previewDisplayed() const
    {
        return preview_ && preview_->displayed;
    }
    bool hasPreview() const
    {
        return bool( preview_ );
    }
    const std::optional<Solution> &result() const
    {
        return result_;
    }

private:
    void watchGeometry_();
    Expected<void> cancelPreview_( bool abandonChangedPreview = false );
    bool canCancelPreview_() const;
    ObjectPtr root_;
    TransformHistory &history_;
    Selection selection_;
    Settings settings_;
    struct Preview
    {
        SceneStamp scene;
        AffineXf3f target;
        uint64_t geometryRevision{};
        ObjectPtr moving;
        const Object *parent{};
        AffineXf3f before;
        bool displayed{};
        std::vector<NodeStamp> subtree;
    };
    struct Restore
    {
        SceneStamp scene;
        ObjectPtr moving;
        AffineXf3f before;
        uint64_t geometryRevision{};
    };
    std::optional<Preview> preview_;
    std::optional<Restore> restore_;
    std::optional<Solution> result_;
    std::shared_ptr<uint64_t> geometryRevision_{ std::make_shared<uint64_t>( 0 ) };
    std::vector<boost::signals2::scoped_connection> connections_;
};
} // namespace MR::FeatureAlignmentTool
