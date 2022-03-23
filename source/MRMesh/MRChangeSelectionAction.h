#pragma once
#include "MRHistoryAction.h"
#include "MRObjectMesh.h"

namespace MR
{

// Undo action for ObjectMesh face selection
class ChangeSelectionAction : public HistoryAction
{
public:
    // use this constructor to remember object's face selection before making any changes in it
    ChangeSelectionAction( const std::string& name, const std::shared_ptr<ObjectMesh>& objMesh ):
        name_{name},
        objMesh_{objMesh}
    {
        if ( !objMesh_ )
            return; 
        selection_ = objMesh_->getSelectedFaces();
    }

    // use this constructor to remember current object's face selection and immediately set new selection to the object
    ChangeSelectionAction( const std::string& name, const std::shared_ptr<ObjectMesh>& objMesh, FaceBitSet newSelection ) :
        name_{name},
        objMesh_{objMesh}
    {
        if ( !objMesh_ )
            return;
        selection_ = objMesh_->getSelectedFaces();
        objMesh_->selectFaces( std::move( newSelection ) );
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

// Undo action for ObjectMesh edge selection
class ChangeEdgeSelectionAction : public HistoryAction
{
public:
    // use this constructor to remember object's edge selection before making any changes in it
    ChangeEdgeSelectionAction( const std::string& name, const std::shared_ptr<ObjectMesh>& objMesh ) :
        name_{ name },
        objMesh_{ objMesh }
    {
        if( !objMesh_ )
            return;
        selection_ = objMesh_->getSelectedEdges();
    }

    // use this constructor to remember current object's edge selection and immediately set new selection to the object
    ChangeEdgeSelectionAction( const std::string& name, const std::shared_ptr<ObjectMesh>& objMesh, UndirectedEdgeBitSet newSelection ) :
        name_{name},
        objMesh_{objMesh}
    {
        if ( !objMesh_ )
            return;
        selection_ = objMesh_->getSelectedEdges();
        objMesh_->selectEdges( std::move( newSelection ) );
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

// Undo action for ObjectMesh creases
class ChangeCreasesAction : public HistoryAction
{
public:
    // use this constructor to remember object's creases before making any changes in it
    ChangeCreasesAction( const std::string& name, const std::shared_ptr<ObjectMesh>& objMesh ) :
        name_{ name },
        objMesh_{ objMesh }
    {
        if( !objMesh_ )
            return;
        creases_ = objMesh_->creases();
    }

    // use this constructor to remember current object's creases and immediately set new creases to the object
    ChangeCreasesAction( const std::string& name, const std::shared_ptr<ObjectMesh>& objMesh, UndirectedEdgeBitSet newCreases ) :
        name_{name},
        objMesh_{objMesh}
    {
        if ( !objMesh_ )
            return;
        creases_ = objMesh_->creases();
        objMesh_->setCreases( std::move( newCreases ) );
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