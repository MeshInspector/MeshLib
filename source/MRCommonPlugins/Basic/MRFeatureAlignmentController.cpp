#include "MRFeatureAlignmentController.h"
#include <MRViewer/MRAppendHistory.h>
#include <MRViewer/MRI18n.h>
#include <MRMesh/MRChangeXfAction.h>
#include <MRMesh/MRCombinedHistoryAction.h>
#include <MRMesh/MRFinally.h>
#include <MRMesh/MRObjectMesh.h>
#include <MRMesh/MRObjectPoints.h>
#include <MRMesh/MRObjectLines.h>
#include <algorithm>
#include <unordered_set>

namespace MR::FeatureAlignmentTool
{
Controller::Controller( ObjectPtr root ) : root_( std::move( root ) ) {}
Controller::~Controller()
{
    (void)cancelPreview_( true );
}
void Controller::refresh()
{
    viewDirty_ = true;
}
void Controller::selectedChanged_()
{
    if ( changingPose_ )
        return;
    viewDirty_ = true;
    previewChanged_ = bool( preview_ );
}
void Controller::watchHistory_()
{
    auto *current = HistoryStore::getViewerInstance();
    if ( current == history_ && ( !current || historyConnection_.connected() ) )
        return;
    historyConnection_.disconnect();
    history_ = current;
    refresh();
    if ( history_ )
        historyConnection_ = history_->changedSignal.connect(
            [this]( HistoryStore &store, HistoryStore::ChangeType change, std::shared_ptr<HistoryAction> action )
            {
                if ( changingPose_ )
                    return;
                // Wait until a native scope is complete before removing Preview.
                if ( ( change == HistoryStore::ChangeType::PreAppendAction ||
                       change == HistoryStore::ChangeType::PostAppendAction ) && store.getScopeBlockPtr() )
                {
                    refresh();
                    return;
                }
                auto *group = change == HistoryStore::ChangeType::PreAppendAction
                    ? dynamic_cast<CombinedHistoryAction *>( action.get() ) : nullptr;
                (void)cancelPreview_( true, group );
                result_.reset();
                refresh();
            } );
}
void Controller::watchSelected_()
{
    std::vector<ObjectPtr> objects, pending;
    if ( selection_.fixed )
        pending.push_back( selection_.fixed );
    if ( selection_.moving )
        pending.push_back( selection_.moving );
    std::unordered_set<const Object *> visited;
    while ( !pending.empty() )
    {
        auto object = std::move( pending.back() );
        pending.pop_back();
        if ( !visited.insert( object.get() ).second )
            continue;
        objects.push_back( object );
        for ( const auto &child : object->children() )
            pending.push_back( child );
    }
    if ( objects == watchedObjects_ )
        return;
    connections_.clear();
    watchedObjects_ = std::move( objects );
    for ( const auto &object : watchedObjects_ )
    {
        connections_.emplace_back( object->worldXfChangedSignal.connect( [this] { selectedChanged_(); } ) );
        const auto changed = [this]( uint32_t ) { selectedChanged_(); };
        if ( const auto mesh = std::dynamic_pointer_cast<ObjectMesh>( object ) )
            connections_.emplace_back( mesh->meshChangedSignal.connect( changed ) );
        else if ( const auto points = std::dynamic_pointer_cast<ObjectPoints>( object ) )
        {
            connections_.emplace_back( points->pointsChangedSignal.connect( changed ) );
            connections_.emplace_back( points->normalsChangedSignal.connect( changed ) );
        }
        else if ( const auto lines = std::dynamic_pointer_cast<ObjectLines>( object ) )
            connections_.emplace_back( lines->linesChangedSignal.connect( changed ) );
    }
}
const Controller::ViewState &Controller::viewState()
{
    watchHistory_();
    if ( !viewDirty_ )
    {
        viewState_.stalePreview = stalePreview();
        viewState_.canApply = canApply();
        return viewState_;
    }
    viewState_.entries = catalog( root_ );
    for ( auto *options : { &viewState_.primaryFeatures, &viewState_.locatorFeatures, &viewState_.clockingFeatures } )
        for ( auto &side : *options )
            side.clear();
    for ( const auto &entry : viewState_.entries )
        for ( int side = 0; side < 2; ++side )
        {
            const auto &part = side ? selection_.moving : selection_.fixed;
            if ( !part || !below( entry.object.get(), part.get() ) ||
                 ( !side && below( entry.object.get(), selection_.moving.get() ) ) )
                continue;
            const bool primary = settings_.mode == PrimaryMode::Plane ? isPlane( entry.object )
                               : settings_.mode == PrimaryMode::Axis ? isAxis( entry.object ) : isCenter( entry.object );
            if ( primary )
                viewState_.primaryFeatures[side].push_back( entry );
            if ( settings_.mode == PrimaryMode::Plane ? isHole( entry.object ) : isCenter( entry.object ) )
                viewState_.locatorFeatures[side].push_back( entry );
            if ( isDirection( entry.object ) )
                viewState_.clockingFeatures[side].push_back( entry );
        }
    const auto ready = readConstraints( root_, viewState_.entries, selection_, settings_ );
    viewState_.canPreview = bool( ready );
    viewState_.setupError = ready ? std::string() : ready.error();
    viewState_.stalePreview = stalePreview();
    viewState_.canApply = canApply();
    viewState_.canSwapRoles = bool( validateSwap_( viewState_.entries ) );
    watchSelected_();
    viewDirty_ = false;
    return viewState_;
}
Expected<void> Controller::configure( Selection selection, Settings settings )
{
    auto canceled = invalidate();
    if ( !canceled )
        return canceled;
    selection_ = std::move( selection );
    settings_ = settings;
    watchSelected_();
    watchHistory_();
    return {};
}
Expected<void> Controller::validateSwap_( const Catalog &entries ) const
{
    const auto partPresent = [&]( const ObjectPtr &part )
    {
        return part && part != root_ && std::any_of( entries.begin(), entries.end(),
            [&]( const CatalogEntry &entry ) { return entry.object == part && entry.part; } );
    };
    if ( selection_.fixed == selection_.moving || !partPresent( selection_.fixed ) || !partPresent( selection_.moving ) )
        return unexpected( std::string( _t( "Select two valid parts before swapping roles." ) ) );
    if ( below( selection_.moving.get(), selection_.fixed.get() ) )
        return unexpected( std::string( _t( "Cannot swap: the new Floating object would contain the Reference object." ) ) );
    if ( hasPreview() && ( preview_->moving->xf() != preview_->target ||
                                preview_->moving->parent() != preview_->parent ) )
        return unexpected( std::string( _t( "The preview has newer edits. Close the tool to keep them before swapping roles." ) ) );
    return {};
}
Expected<void> Controller::swapRoles()
{
    const auto entries = catalog( root_ );
    auto valid = validateSwap_( entries );
    if ( !valid )
        return valid;
    auto next = selection_;
    std::swap( next.fixed, next.moving );
    std::swap( next.fixedPrimary, next.movingPrimary );
    std::swap( next.fixedLocators, next.movingLocators );
    std::swap( next.fixedClock, next.movingClock );
    const auto ordered = hierarchyRoles( entries, next.fixed, next.moving );
    if ( !ordered )
        return unexpected( ordered.error() );
    next.explicitRoles = ordered->fixed != next.fixed;
    return configure( std::move( next ), settings_ );
}
Expected<void> Controller::cancelPreview_( bool abandonChangedPreview, CombinedHistoryAction *pendingGroup )
{
    if ( hasPreview() )
    {
        if ( preview_->moving->xf() == preview_->target && preview_->moving->parent() == preview_->parent &&
             below( preview_->moving.get(), root_.get() ) )
        {
            std::vector<std::pair<ObjectPtr, AffineXf3f>> incoming;
            std::vector<ObjectPtr> pending{ preview_->moving };
            while ( !pending.empty() )
            {
                auto object = std::move( pending.back() );
                pending.pop_back();
                if ( !preview_->members.contains( object ) )
                    incoming.emplace_back( object, object->worldXf() );
                else
                    for ( const auto &child : object->children() )
                        pending.push_back( child );
            }
            changingPose_ = true;
            MR_FINALLY { changingPose_ = false; };
            preview_->moving->setXf( preview_->before );
            // Native reparenting preserves world poses before appending history.
            // Preserve newly attached branches when their temporary parent resets.
            for ( const auto &[object, world] : incoming )
                if ( object->worldXf() != world )
                {
                    const auto correction = pendingGroup ? std::make_shared<ChangeXfAction>(
                        s_tr( "Preserve object placement" ), object ) : nullptr;
                    object->setWorldXf( world );
                    // Undo this correction before the original native reorder.
                    if ( correction )
                        pendingGroup->getStack().push_back( correction );
                }
        }
        else if ( !abandonChangedPreview )
            return unexpected( std::string( _t( "The Floating object changed outside Preview. Close the tool to keep those edits." ) ) );
    }
    preview_.reset();
    previewChanged_ = false;
    return {};
}
Expected<void> Controller::invalidate( bool abandonChangedPreview )
{
    refresh();
    auto canceled = cancelPreview_( abandonChangedPreview );
    if ( !canceled )
        return canceled;
    result_.reset();
    return {};
}
bool Controller::stalePreview() const
{
    return preview_ && ( previewChanged_ || preview_->moving->parent() != preview_->parent ||
        preview_->moving->xf() != preview_->target ||
        !below( preview_->moving.get(), root_.get() ) || below( selection_.fixed.get(), preview_->moving.get() ) );
}
bool Controller::canApply() const
{
    const auto *history = HistoryStore::getViewerInstance();
    return preview_ && result_ && result_->acceptable() && !stalePreview() && history && !history->undoRedoInProgress();
}
Expected<Solution> Controller::preview()
{
    auto canceled = invalidate();
    if ( !canceled )
        return unexpected( canceled.error() );
    auto constraints = readConstraints( root_, selection_, settings_ );
    if ( !constraints )
        return unexpected( constraints.error() );
    auto solution = FeatureAlignment::solve( *constraints );
    if ( !solution )
        return unexpected( solution.error() );
    const auto parent = selection_.moving->parent() ? AffineXf3d( selection_.moving->parent()->worldXf() ) : AffineXf3d{};
    const auto world = AffineXf3d( selection_.moving->worldXf() );
    const auto target = AffineXf3f( parent.inverse() * solution->delta * world );
    auto actual = FeatureAlignment::measure( *constraints, parent * AffineXf3d( target ) * world.inverse() );
    if ( !actual )
        return unexpected( actual.error() );
    actual->remainingDof = solution->remainingDof;
    actual->rotationSource = solution->rotationSource;
    actual->yawDeg = solution->yawDeg;
    actual->freeMotion = solution->freeMotion;
    for ( const auto &warning : solution->warnings )
        if ( std::find( actual->warnings.begin(), actual->warnings.end(), warning ) == actual->warnings.end() )
            actual->warnings.push_back( warning );
    result_ = *actual;
    preview_ = Preview{ selection_.moving, selection_.moving->parent(), selection_.moving->xf(), target, {} };
    std::vector<ObjectPtr> pending{ selection_.moving };
    while ( !pending.empty() )
    {
        auto object = std::move( pending.back() );
        pending.pop_back();
        preview_->members.insert( object );
        for ( const auto &child : object->children() )
            pending.push_back( child );
    }
    changingPose_ = true;
    MR_FINALLY { changingPose_ = false; };
    selection_.moving->setXf( target );
    return *result_;
}
Expected<void> Controller::apply()
{
    if ( !preview_ || !result_ )
        return unexpected( std::string( _t( "Preview the current setup before applying." ) ) );
    if ( stalePreview() )
        return unexpected( std::string( _t( "The selected parts changed after Preview. Preview again before applying." ) ) );
    if ( !result_->acceptable() )
        return unexpected( std::string( _t( "Resolve the reported measurement conflicts before applying." ) ) );
    const auto checked = readConstraints( root_, selection_, settings_ );
    if ( !checked )
        return unexpected( checked.error() );
    const auto *history = HistoryStore::getViewerInstance();
    if ( !history || history->undoRedoInProgress() )
        return unexpected( std::string( _t( "Native undo history is unavailable. The placement was not applied." ) ) );
    const auto placement = *preview_;
    preview_.reset();
    previewChanged_ = false;
    changingPose_ = true;
    MR_FINALLY { changingPose_ = false; };
    placement.moving->setXf( placement.before );
    AppendHistory<ChangeXfAction>( s_tr( "Feature alignment" ), placement.moving, placement.target );
    refresh();
    return {};
}
} // namespace MR::FeatureAlignmentTool
