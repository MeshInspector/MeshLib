#pragma once
#include "MRHistoryAction.h"
#include "MRObjectMesh.h"

namespace MR
{

class ChangeSelectionAction : public HistoryAction
{
public:
    ChangeSelectionAction( const std::string& name, const std::shared_ptr<ObjectMesh>& objMesh ):
        name_{name},
        objMesh_{objMesh}
    {
        if ( !objMesh_ )
            return; 
        selection_ = objMesh_->getSelectedFaces();
    }

    virtual std::string name() const override { return name_; }

    virtual void action( Type ) override
    {
        if ( !objMesh_ )
            return;
        auto tmp = objMesh_->getSelectedFaces();
        objMesh_->selectFaces( selection_ );
        selection_ = std::move( tmp );
    }

private:
    std::string name_;
    std::shared_ptr<ObjectMesh> objMesh_;
    FaceBitSet selection_;
};

class ChangeEdgeSelectionAction : public HistoryAction
{
public:
    ChangeEdgeSelectionAction( const std::string& name, const std::shared_ptr<ObjectMesh>& objMesh ) :
        name_{ name },
        objMesh_{ objMesh }
    {
        if( !objMesh_ )
            return;
        selection_ = objMesh_->getSelectedEdges();
    }

    virtual std::string name() const override { return name_; }

    virtual void action( Type ) override
    {
        if( !objMesh_ )
            return;
        auto tmp = objMesh_->getSelectedEdges();
        objMesh_->selectEdges( selection_ );
        selection_ = std::move( tmp );
    }

private:
    std::string name_;
    std::shared_ptr<ObjectMesh> objMesh_;
    UndirectedEdgeBitSet selection_;
};

}