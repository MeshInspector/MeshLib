#include "MRFeatureAlignmentController.h"
#include <MRMesh/MRObjectMesh.h>
#include <MRMesh/MRObjectPoints.h>
#include <MRMesh/MRObjectLines.h>
#include <algorithm>

namespace MR::FeatureAlignmentTool
{
Expected<void> TransformHistory::commitPreview( const ObjectPtr &object, const AffineXf3f &before,
                                                const AffineXf3f &target, const std::string &label )
{
    // One GUI-thread operation: history captures the original pose, then the
    // target is restored before another frame can be rendered.
    const auto current = object->xf();
    object->setXf( before );
    try
    {
        auto result = commit( object, target, label );
        if ( !result )
            object->setXf( current );
        return result;
    }
    catch ( ... )
    {
        object->setXf( current );
        throw;
    }
}
Controller::Controller( ObjectPtr root, TransformHistory &history ) : root_( std::move( root ) ), history_( history )
{
    watchGeometry_();
}
Controller::~Controller()
{
    // The native panel destroys its controller on the GUI thread.
    (void)cancelPreview_();
}
Expected<void> Controller::configure( Selection selection, Settings settings )
{
    auto canceled = invalidate();
    if ( !canceled )
        return canceled;
    if ( selection.moving != selection_.moving )
        restore_.reset();
    selection_ = std::move( selection );
    settings_ = settings;
    watchGeometry_();
    return {};
}
bool Controller::canCancelPreview_() const
{
    if ( !preview_ || !preview_->displayed || !preview_->moving || !below( preview_->moving.get(), root_.get() ) ||
         preview_->moving->parent() != preview_->parent || preview_->moving->xf() != preview_->target ||
         ( selection_.fixed && below( selection_.fixed.get(), preview_->moving.get() ) ) )
        return false;
    size_t count = 0;
    for ( const auto &entry : catalog( root_ ) )
        if ( below( entry.object.get(), preview_->moving.get() ) )
        {
            ++count;
            const auto original = std::find_if( preview_->subtree.begin(), preview_->subtree.end(),
                                                [&]( const NodeStamp &node ) { return node.object == entry.object; } );
            if ( original == preview_->subtree.end() || entry.object->parent() != original->parent )
                return false;
        }
    return count == preview_->subtree.size();
}
Expected<void> Controller::cancelPreview_( bool abandonChangedPreview )
{
    if ( preview_ && preview_->displayed )
    {
        if ( !canCancelPreview_() )
        {
            if ( !abandonChangedPreview )
                return unexpected( std::string( "The Floating obj pose or hierarchy changed outside Preview. "
                                                "Close the tool to keep those edits, or undo them before Restore." ) );
            // Closing releases stale preview ownership without any scene write.
        }
        else
            preview_->moving->setXf( preview_->before );
    }
    preview_.reset();
    return {};
}
Expected<void> Controller::invalidate( bool abandonChangedPreview )
{
    auto canceled = cancelPreview_( abandonChangedPreview );
    if ( !canceled )
        return canceled;
    result_.reset();
    return {};
}
void Controller::watchGeometry_()
{
    connections_.clear();
    for ( const auto &e : catalog( root_ ) )
        if ( auto mesh = std::dynamic_pointer_cast<ObjectMesh>( e.object ) )
        {
            const auto revision = geometryRevision_;
            connections_.emplace_back( mesh->meshChangedSignal.connect(
                [revision]( uint32_t mask )
                {
                    if ( mask & ( DIRTY_POSITION | DIRTY_FACE ) )
                        ++*revision;
                } ) );
        }
        else if ( auto points = std::dynamic_pointer_cast<ObjectPoints>( e.object ) )
        {
            const auto revision = geometryRevision_;
            connections_.emplace_back( points->pointsChangedSignal.connect( [revision]( uint32_t ) { ++*revision; } ) );
            connections_.emplace_back(
                points->normalsChangedSignal.connect( [revision]( uint32_t ) { ++*revision; } ) );
        }
        else if ( auto lines = std::dynamic_pointer_cast<ObjectLines>( e.object ) )
        {
            const auto revision = geometryRevision_;
            connections_.emplace_back( lines->linesChangedSignal.connect( [revision]( uint32_t ) { ++*revision; } ) );
        }
}
bool Controller::stalePreview() const
{
    if ( !preview_ )
        return false;
    auto matches = preview_->scene.matches( root_ );
    return preview_->geometryRevision != *geometryRevision_ || !matches || !*matches;
}
bool Controller::canApply() const
{
    return preview_ && result_ && result_->acceptable() && !stalePreview();
}
bool Controller::canRestore() const
{
    if ( previewDisplayed() )
        return canCancelPreview_();
    if ( !restore_ || restore_->geometryRevision != *geometryRevision_ )
        return false;
    auto matches = restore_->scene.matches( root_ );
    return matches && *matches;
}
Expected<Solution> Controller::preview( bool show )
{
    auto canceled = invalidate();
    if ( !canceled )
        return unexpected( canceled.error() );
    auto constraints = readConstraints( root_, selection_, settings_ );
    if ( !constraints )
        return unexpected( constraints.error() );
    auto before = stamp( root_ );
    if ( !before )
        return unexpected( before.error() );
    auto solution = FeatureAlignment::solve( *constraints );
    if ( !solution )
        return unexpected( solution.error() );
    const auto parent =
        selection_.moving->parent() ? AffineXf3d( selection_.moving->parent()->worldXf() ) : AffineXf3d{};
    const auto world = AffineXf3d( selection_.moving->worldXf() );
    const auto local = parent.inverse() * solution->delta * world;
    const auto target = AffineXf3f( local );
    // Validate the actual single-precision transform before making any write.
    const auto actualDelta = parent * AffineXf3d( target ) * world.inverse();
    auto actual = FeatureAlignment::measure( *constraints, actualDelta );
    if ( !actual )
        return unexpected( actual.error() );
    actual->remainingDof = solution->remainingDof;
    actual->rotationSource = solution->rotationSource;
    actual->yawDeg = solution->yawDeg;
    actual->freeMotion = solution->freeMotion;
    if ( !actual->acceptable() && actual->remainingDof == 0 )
        actual->freeMotion = "No motion remains, but the selected measurements conflict.";
    for ( const auto &warning : solution->warnings )
        if ( std::find( actual->warnings.begin(), actual->warnings.end(), warning ) == actual->warnings.end() )
            actual->warnings.push_back( warning );
    result_ = *actual;
    preview_ = Preview{
        *before, target, *geometryRevision_, selection_.moving, selection_.moving->parent(), selection_.moving->xf(),
        false };
    if ( show )
    {
        selection_.moving->setXf( target );
        auto displayed = stamp( root_ );
        if ( !displayed )
        {
            selection_.moving->setXf( preview_->before );
            preview_.reset();
            return unexpected( displayed.error() );
        }
        preview_->scene = *displayed;
        preview_->displayed = true;
        for ( const auto &node : displayed->nodes )
            if ( below( node.object.get(), selection_.moving.get() ) )
                preview_->subtree.push_back( node );
    }
    watchGeometry_();
    return *result_;
}
Expected<void> Controller::apply()
{
    if ( !preview_ || !result_ )
        return unexpected( std::string( "Preview the current setup before applying." ) );
    if ( stalePreview() )
        return unexpected( std::string( "The scene changed after Preview. Preview again before applying." ) );
    if ( !result_->acceptable() )
        return unexpected( std::string( "Resolve the reported measurement conflicts before applying." ) );
    // Ownership is revalidated immediately before the only transform write.
    auto checked = readConstraints( root_, selection_, settings_ );
    if ( !checked )
        return unexpected( checked.error() );
    const auto before = preview_->before;
    auto committed = preview_->displayed
                         ? history_.commitPreview( selection_.moving, before, preview_->target, "Feature alignment" )
                         : history_.commit( selection_.moving, preview_->target, "Feature alignment" );
    if ( !committed )
        return unexpected( committed.error() );
    auto after = stamp( root_ );
    if ( !after )
        return unexpected( after.error() );
    restore_ = Restore{ *after, selection_.moving, before, *geometryRevision_ };
    preview_.reset();
    watchGeometry_();
    return {};
}
Expected<void> Controller::restore()
{
    if ( previewDisplayed() )
        return invalidate();
    if ( !restore_ )
        return unexpected( std::string( "There is no placement to restore in this session." ) );
    if ( !canRestore() )
        return unexpected(
            std::string( "The scene changed after Apply. Use native Undo, or inspect the scene before restoring." ) );
    auto committed = history_.commit( restore_->moving, restore_->before, "Restore feature alignment" );
    if ( !committed )
        return unexpected( committed.error() );
    restore_.reset();
    (void)invalidate();
    watchGeometry_();
    return {};
}
} // namespace MR::FeatureAlignmentTool
