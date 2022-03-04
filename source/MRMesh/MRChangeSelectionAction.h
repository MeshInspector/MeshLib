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

    const FaceBitSet & selection() const { return selection_; }

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
        objMesh_->selectEdges( std::move( selection_ ) );
        selection_ = std::move( tmp );
    }

    const UndirectedEdgeBitSet & selection() const { return selection_; }

private:
    std::string name_;
    std::shared_ptr<ObjectMesh> objMesh_;
    UndirectedEdgeBitSet selection_;
};

class ChangeCreasesAction : public HistoryAction
{
public:
    ChangeCreasesAction( const std::string& name, const std::shared_ptr<ObjectMesh>& objMesh ) :
        name_{ name },
        objMesh_{ objMesh }
    {
        if( !objMesh_ )
            return;
        creases_ = objMesh_->creases();
    }

    virtual std::string name() const override { return name_; }

    virtual void action( Type ) override
    {
        if( !objMesh_ )
            return;
        auto tmp = objMesh_->creases();
        objMesh_->setCreases( std::move( creases_ ) );
        creases_ = std::move( tmp );
    }

    const UndirectedEdgeBitSet & creases() const { return creases_; }

private:
    std::string name_;
    std::shared_ptr<ObjectMesh> objMesh_;
    UndirectedEdgeBitSet creases_;
};

}