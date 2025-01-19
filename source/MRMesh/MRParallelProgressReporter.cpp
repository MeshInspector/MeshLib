#include "MRParallelProgressReporter.h"


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
    progress_ += delta / static_cast<float>( totalWeight_ );
    if ( mainThreadId_ == std::this_thread::get_id() )
        return (*this)();
    return continue_;
}



bool ParallelProgressReporter::PerTaskReporter::operator()( float p ) const
{
    bool res = reporter_->updateTask_( ( p - task_->progress ) * task_->weight );
    task_->progress = p;
    return res;
}


}