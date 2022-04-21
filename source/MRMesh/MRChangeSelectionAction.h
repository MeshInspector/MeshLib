#pragma once
#include "MRHistoryAction.h"
#include "MRObjectMesh.h"
#include "MRObjectPoints.h"

namespace MR
{

// Undo action for ObjectMesh face selection
class ChangeMeshFaceSelectionAction : public HistoryAction
{
public:
    using Obj = ObjectMesh;

    // use this constructor to remember object's face selection before making any changes in it
    ChangeMeshFaceSelectionAction( const std::string& name, const std::shared_ptr<ObjectMesh>& objMesh ):
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

    // empty because set dirty is inside selectFaces
    static void setObjectDirty( const std::shared_ptr<ObjectMesh>& ) {}

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + selection_.heapBytes();
    }

private:
    std::string name_;
    std::shared_ptr<ObjectMesh> objMesh_;
    FaceBitSet selection_;
};

// Undo action for ObjectMesh edge selection
class ChangeMeshEdgeSelectionAction : public HistoryAction
{
public:
    using Obj = ObjectMesh;

    // use this constructor to remember object's edge selection before making any changes in it
    ChangeMeshEdgeSelectionAction( const std::string& name, const std::shared_ptr<ObjectMesh>& objMesh ) :
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

    // empty because set dirty is inside selectEdges
    static void setObjectDirty( const std::shared_ptr<ObjectMesh>& ) {}

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + selection_.heapBytes();
    }

private:
    std::string name_;
    std::shared_ptr<ObjectMesh> objMesh_;
    UndirectedEdgeBitSet selection_;
};

// Undo action for ObjectMesh creases
class ChangeMeshCreasesAction : public HistoryAction
{
public:
    using Obj = ObjectMesh;

    // use this constructor to remember object's creases before making any changes in it
    ChangeMeshCreasesAction( const std::string& name, const std::shared_ptr<ObjectMesh>& objMesh ) :
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

    // empty because set dirty is inside setCreases
    static void setObjectDirty( const std::shared_ptr<ObjectMesh>& ) {}

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + creases_.heapBytes();
    }

private:
    std::string name_;
    std::shared_ptr<ObjectMesh> objMesh_;
    UndirectedEdgeBitSet creases_;
};

// Undo action for ObjectPoints point selection
class ChangePointPointSelectionAction : public HistoryAction
{
public:
    // use this constructor to remember object's vertex selection before making any changes in it
    ChangePointPointSelectionAction( const std::string& name, const std::shared_ptr<ObjectPoints>& objPoints ) :
        name_{ name },
        objPoints_{ objPoints }
    {
        if ( !objPoints_ )
            return;
        selection_ = objPoints_->getSelectedVertices();
    }

    virtual std::string name() const override { return name_; }

    virtual void action( Type ) override
    {
        if ( !objPoints_ )
            return;
        auto tmp = objPoints_->getSelectedVertices();
        objPoints_->selectVertices( selection_ );
        selection_ = std::move( tmp );
    }

    const VertBitSet& selection() const { return selection_; }

    // empty because set dirty is inside selectPoints
    static void setObjectDirty( const std::shared_ptr<ObjectPoints>& )
    {}

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + selection_.heapBytes();
    }

private:
    std::string name_;
    std::shared_ptr<ObjectPoints> objPoints_;
    VertBitSet selection_;
};
}