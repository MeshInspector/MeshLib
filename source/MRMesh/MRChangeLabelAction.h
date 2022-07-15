#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "MRHistoryAction.h"
#include "MRMesh/MRPositionedText.h"
#include "MRMesh/MRVector2.h"

namespace MR
{

class ChangeLabelAction : public HistoryAction
{
public:
    ChangeLabelAction( const std::string& actionName, std::shared_ptr<ObjectLabel> obj ) :
        obj_( std::move( obj ) ),
        actionName_( actionName )
    {
        label_ = obj_->getLabel();
        pivotPoint_ = obj_->getPivotPoint();
    }


    virtual std::string name() const override { return actionName_; }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !obj_ )
            return;

        auto label = obj_->getLabel();
        obj_->setLabel( std::move( label_ ) );
        label_ = std::move( label );


        auto pivotPoint = obj_->getPivotPoint();
        obj_->setPivotPoint( std::move( pivotPoint_ ) );
        pivotPoint_ = std::move( pivotPoint );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return label_.text.capacity() + sizeof( float ) * 5 + actionName_.capacity();
    }
private:
    std::shared_ptr<ObjectLabel> obj_;
    PositionedText label_;
    Vector2f pivotPoint_;

    std::string actionName_;
};

}
