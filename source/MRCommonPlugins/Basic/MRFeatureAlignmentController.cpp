#include "MRFeatureAlignmentController.h"
#include <MRViewer/MRAppendHistory.h>
#include <MRViewer/MRI18n.h>
#include <MRMesh/MRChangeXfAction.h>
#include <MRMesh/MRFeatureObject.h>
#include <MRMesh/MRObjectMesh.h>
#include <MRMesh/MRObjectPoints.h>
#include <MRMesh/MRObjectLines.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <unordered_set>

namespace MR::FeatureAlignmentTool
{
Controller::Controller( ObjectPtr root ) : root_( std::move( root ) ) {}
void Controller::refresh()
{
    viewDirty_ = true;
}
void Controller::selectedChanged_()
{
    refresh();
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
            [this]( HistoryStore &, HistoryStore::ChangeType, std::shared_ptr<HistoryAction> )
            {
                refresh();
            } );
}
void Controller::watchSelected_()
{
    std::vector<WatchedObject> objects;
    std::vector<ObjectPtr> pending;
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
        objects.push_back( { object, object->parent(), object->children().size() } );
        for ( const auto &child : object->children() )
            pending.push_back( child );
    }
    if ( objects == watchedObjects_ )
        return;
    connections_.clear();
    watchedObjects_ = std::move( objects );
    for ( const auto &entry : watchedObjects_ )
    {
        const auto &object = entry.object;
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
    // addChild signals the new child, and removal can be silent. Check only the branches already being watched.
    if ( !viewDirty_ && std::none_of( watchedObjects_.begin(), watchedObjects_.end(), []( const WatchedObject &entry )
        { return entry.object->parent() != entry.parent || entry.object->children().size() != entry.childCount; } ) )
    {
        viewState_.canApply = canApply_();
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
    result_.reset();
    const auto target = evaluate_( viewState_.entries );
    viewState_.setupError = target ? std::string() : target.error();
    viewState_.canApply = canApply_();
    viewState_.canSwapRoles = bool( validateSwap_( viewState_.entries ) );
    watchSelected_();
    viewDirty_ = false;
    return viewState_;
}
void Controller::configure( Selection selection, Settings settings )
{
    result_.reset();
    refresh();
    selection_ = std::move( selection );
    settings_ = settings;
    watchSelected_();
    watchHistory_();
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
    configure( std::move( next ), settings_ );
    return {};
}
bool Controller::canApply_() const
{
    const auto *history = HistoryStore::getViewerInstance();
    return result_ && result_->acceptable() && selection_.moving && !selection_.moving->isLocked() &&
           history && !history->undoRedoInProgress() && !history->getScopeBlockPtr();
}
Expected<AffineXf3f> Controller::evaluate_( const Catalog &entries )
{
    auto refs = readReferences( root_, entries, selection_ );
    if ( !refs )
        return unexpected( refs.error() );
    auto constraints = FeatureAlignment::readConstraints( *refs, settings_ );
    if ( !constraints )
        return unexpected( constraints.error() );
    const auto solution = FeatureAlignment::solve( *constraints );
    if ( !solution )
        return unexpected( solution.error() );
    const auto parent = selection_.moving->parent() ? AffineXf3d( selection_.moving->parent()->worldXf() ) : AffineXf3d{};
    const auto world = AffineXf3d( selection_.moving->worldXf() );
    const auto target = AffineXf3f( parent.inverse() * solution->delta * world );
    // Match Object::worldXf's float operations and grouping, including intermediate feature parents.
    const auto worldAtTarget = [&]( const Object *node )
    {
        auto xf = node == selection_.moving.get() ? target : node->xf();
        for ( node = node->parent(); node; node = node->parent() )
            xf = ( node == selection_.moving.get() ? target : node->xf() ) * xf;
        return xf;
    };
    const auto roundedDelta = AffineXf3d( worldAtTarget( selection_.moving.get() ) ) * world.inverse();
    const std::array pairs{ &refs->primary, &refs->locators[0], &refs->locators[1], &refs->clocking };
    const bool nested = std::any_of( pairs.begin(), pairs.end(), [&]( const auto *pair )
    {
        return pair->moving && pair->moving->parent() != selection_.moving.get();
    } );
    if ( nested )
    {
        // Intermediate parents round independently in Object::worldXf. Measure their actual float frames.
        std::array<Object, 4> frames;
        for ( size_t i = 0; i < pairs.size(); ++i )
        {
            auto &moving = pairs[i]->moving;
            if ( !moving )
                continue;
            const auto xf = worldAtTarget( moving->parent() );
            const auto det = xf.A.det();
            if ( !( det > 0 ) || !std::isfinite( det ) )
                return unexpected( std::string( _t( "The proposed feature transform is not rigid." ) ) );
            frames[i].setXf( xf );
            auto copy = std::static_pointer_cast<FeatureObject>( moving->shallowClone() );
            frames[i].addChild( copy );
            moving = std::move( copy );
        }
        constraints = FeatureAlignment::readConstraints( *refs, settings_ );
        if ( !constraints )
            return unexpected( constraints.error() );
    }
    auto actual = FeatureAlignment::measure( *constraints, nested ? AffineXf3d{} : roundedDelta );
    if ( !actual )
        return unexpected( actual.error() );
    if ( nested )
        actual->delta = roundedDelta;
    for ( const auto &warning : solution->warnings )
        if ( std::find( actual->warnings.begin(), actual->warnings.end(), warning ) == actual->warnings.end() )
            actual->warnings.push_back( warning );
    result_ = std::move( *actual );
    return target;
}
Expected<void> Controller::apply()
{
    if ( selection_.moving && selection_.moving->isLocked() )
        return unexpected( std::string( _t( "Unlock the Floating object's transform before applying." ) ) );
    const auto *history = HistoryStore::getViewerInstance();
    if ( !history || history->undoRedoInProgress() || history->getScopeBlockPtr() )
        return unexpected( std::string( _t( "Native undo history is unavailable. The placement was not applied." ) ) );
    // Re-read immediately before Apply; cached dialog results never authorize a stale placement.
    result_.reset();
    refresh();
    const auto target = evaluate_( catalog( root_ ) );
    if ( !target )
        return unexpected( target.error() );
    if ( !result_->acceptable() )
        return unexpected( std::string( _t( "Resolve the reported measurement conflicts before applying." ) ) );
    AppendHistory<ChangeXfAction>( s_tr( "Feature alignment" ), selection_.moving, *target );
    return {};
}
} // namespace MR::FeatureAlignmentTool
