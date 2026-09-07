#include "MRParallelProgressReporter.h"
#include <cassert>

namespace MR
{


ParallelProgressReporter::ParallelProgressReporter( const ProgressCallback& cb ) :
    cb_( cb ),
    mainThreadId_( std::this_thread::get_id() )
{}

ParallelProgressReporter::PerTaskReporter ParallelProgressReporter::newTask( float weight )
{
    std::lock_guard lock( mutex_ );
    const float totalWeight = totalWeight_;
    if ( totalWeight + weight > 0 )
        progress_ = progress_ * totalWeight / ( totalWeight + weight );
    totalWeight_ += weight;
    return PerTaskReporter{ .reporter_ = this,
                            .task_ = &perTaskInfo_.emplace_front( TaskInfo{ .progress = 0.f, .weight = weight } ) };
}

bool ParallelProgressReporter::operator()()
{
    std::lock_guard lock( mutex_ );
    return continue_ = cb_( progress_ );
}

bool ParallelProgressReporter::updateTask_( float delta )
{
    std::lock_guard lock( mutex_ );
    assert( delta <= totalWeight_ );
    if ( totalWeight_ > 0 )
        progress_ = std::min( progress_ + delta / static_cast< float >( totalWeight_ ), 1.0f ); // due to float errors it can be 1 + eps, which is asserted in ProgressBar
    if ( mainThreadId_ == std::this_thread::get_id() )
        return continue_ = cb_( progress_ ); // avoid recursive lock here
    return continue_;
}

bool ParallelProgressReporter::PerTaskReporter::operator()( float p ) const
{
    bool res = reporter_->updateTask_( ( p - task_->progress ) * task_->weight );
    task_->progress = p;
    return res;
}

}