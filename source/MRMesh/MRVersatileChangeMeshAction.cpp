#include "MRVersatileChangeMeshAction.h"
#include "MRPartialChangeMeshAction.h"
#include "MRChangeMeshAction.h"

namespace MR
{

VersatileChangeMeshPointsAction::VersatileChangeMeshPointsAction( std::string name, const std::shared_ptr<ObjectMesh>& obj ) :
    stdAction_{ std::make_unique<ChangeMeshPointsAction>( std::move( name ), obj ) }
{
}

VersatileChangeMeshPointsAction::~VersatileChangeMeshPointsAction() = default;

std::string VersatileChangeMeshPointsAction::name() const
{
    return stdAction_ ? stdAction_->name() : diffAction_->name();
}

void VersatileChangeMeshPointsAction::action( HistoryAction::Type t )
{
    if ( stdAction_ )
        stdAction_->action( t );
    else
        diffAction_->action( t );
}

void VersatileChangeMeshPointsAction::setObjectDirty( const std::shared_ptr<ObjectMesh>& obj )
{
    if ( obj )
        obj->setDirtyFlags( DIRTY_POSITION );
}

[[nodiscard]] size_t VersatileChangeMeshPointsAction::heapBytes() const
{
    return MR::heapBytes( stdAction_ ) + MR::heapBytes( diffAction_ );
}

/// switch from uncompressed to compressed format to occupy less amount of memory
void VersatileChangeMeshPointsAction::compress()
{
    assert( stdAction_ );
    if ( stdAction_ )
    {
        diffAction_ = std::make_unique<PartialChangeMeshPointsAction>(
            stdAction_->name(), stdAction_->obj(), cmpOld, stdAction_->clonePoints() );
        stdAction_.reset();
    }
    assert( !stdAction_ );
    assert( diffAction_ );
}

} // namespace MR
