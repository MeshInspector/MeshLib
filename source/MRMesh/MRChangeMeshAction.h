#pragma once
#include "MRHistoryAction.h"
#include "MRObjectMesh.h"
#include "MRMesh.h"
#include "MRHeapBytes.h"
#include <memory>

namespace MR
{

/// \defgroup HistoryGroup History group
/// \{

/// Undo action for ObjectMesh mesh change
class ChangeMeshAction : public HistoryAction
{
public:
    using Obj = ObjectMesh;

    /// use this constructor to remember object's mesh before making any changes in it
    ChangeMeshAction( std::string name, const std::shared_ptr<ObjectMesh>& obj ) :
        objMesh_{ obj },
        name_{ std::move( name ) }
    {
        if ( obj )
        {
            if ( auto m = obj->mesh() )
                cloneMesh_ = std::make_shared<Mesh>( *m );
        }
    }

    /// use this constructor to remember object's mesh and immediately set new mesh
    ChangeMeshAction( std::string name, const std::shared_ptr<ObjectMesh>& obj, std::shared_ptr<Mesh> newMesh ) :
        objMesh_{ obj },
        name_{ std::move( name ) }
    {
        if ( obj )
            cloneMesh_ = objMesh_->updateMesh( std::move( newMesh ) );
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !objMesh_ )
            return;

        cloneMesh_ = objMesh_->updateMesh( cloneMesh_ );
    }

    static void setObjectDirty( const std::shared_ptr<ObjectMesh>& obj )
    {
        if ( obj )
            obj->setDirtyFlags( DIRTY_ALL );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + MR::heapBytes( cloneMesh_ );
    }

private:
    std::shared_ptr<ObjectMesh> objMesh_;
    std::shared_ptr<Mesh> cloneMesh_;

    std::string name_;
};

/// Undo action for ObjectMeshHolder uvCoords change
class ChangeMeshUVCoordsAction : public HistoryAction
{
public:
    using Obj = ObjectMeshHolder;

    /// use this constructor to remember object's uv-coordinates before making any changes in them
    ChangeMeshUVCoordsAction( std::string name, const std::shared_ptr<ObjectMeshHolder>& obj ) :
        objMesh_{ obj },
        name_{ std::move( name ) }
    {
        if ( obj )
        {
            uvCoords_ = obj->getUVCoords();
        }
    }

    /// use this constructor to remember object's uv-coordinates and immediate set new value
    ChangeMeshUVCoordsAction( std::string name, const std::shared_ptr<ObjectMeshHolder>& obj, VertUVCoords&& newUvCoords ) :
        objMesh_{ obj },
        name_{ std::move( name ) }
    {
        if ( obj )
        {
            uvCoords_ = std::move( newUvCoords );
            obj->updateUVCoords( uvCoords_ );
        }
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !objMesh_ )
            return;

        objMesh_->updateUVCoords( uvCoords_ );
    }

    static void setObjectDirty( const std::shared_ptr<ObjectMeshHolder>& obj )
    {
        if ( obj )
            obj->setDirtyFlags( DIRTY_UV );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + uvCoords_.heapBytes();
    }

private:
    VertUVCoords uvCoords_;
    std::shared_ptr<ObjectMeshHolder> objMesh_;
    std::string name_;
};

/// History action for texture change
/// \ingroup HistoryGroup
class ChangeTextureAction : public HistoryAction
{
public:
    using Obj = ObjectMeshHolder;

    /// use this constructor to remember object's textures before making any changes in them
    ChangeTextureAction( std::string name, const std::shared_ptr<ObjectMeshHolder>& obj ) :
        obj_{ obj },
        name_{ std::move( name ) }
    {
        if ( obj )
            textures_ = obj->getTextures();
    }

    /// use this constructor to remember object's textures and immediate set new value
    ChangeTextureAction( std::string name, const std::shared_ptr<ObjectMeshHolder>& obj, Vector<MeshTexture, TextureId>&& newTextures ) :
        obj_{ obj },
        name_{ std::move( name ) }
    {
        if ( obj )
        {
            textures_ = std::move( newTextures );
            obj->updateTextures( textures_ );
        }
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !obj_ )
            return;
        obj_->updateTextures( textures_ );
    }

    static void setObjectDirty( const std::shared_ptr<ObjectMeshHolder>& obj )
    {
        if ( obj )
            obj->setDirtyFlags( DIRTY_TEXTURE );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + MR::heapBytes( textures_ );
    }

private:
    std::shared_ptr<ObjectMeshHolder> obj_;
    Vector<MeshTexture, TextureId> textures_;
    std::string name_;
};

/// Undo action for ObjectMesh points only (not topology) change
class ChangeMeshPointsAction : public HistoryAction
{
public:
    using Obj = ObjectMesh;

    /// use this constructor to remember object's mesh points before making any changes in it
    ChangeMeshPointsAction( std::string name, const std::shared_ptr<ObjectMesh>& obj ) :
        objMesh_{ obj },
        name_{ std::move( name ) }
    {
        if ( !objMesh_ )
            return;
        if ( auto m = objMesh_->mesh() )
            clonePoints_ = m->points;
    }

    /// use this constructor to remember object's mesh points and immediate set new value
    ChangeMeshPointsAction( std::string name, const std::shared_ptr<ObjectMesh>& obj, VertCoords && newCoords ) :
        objMesh_{ obj },
        name_{ std::move( name ) }
    {
        clonePoints_ = std::move( newCoords );
        action( HistoryAction::Type::Redo );
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !objMesh_ )
            return;

        if ( auto m = objMesh_->varMesh() )
        {
            std::swap( m->points, clonePoints_ );
            objMesh_->setDirtyFlags( DIRTY_POSITION );
        }
    }

    static void setObjectDirty( const std::shared_ptr<ObjectMesh>& obj )
    {
        if ( obj )
            obj->setDirtyFlags( DIRTY_POSITION );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + clonePoints_.heapBytes();
    }

private:
    std::shared_ptr<ObjectMesh> objMesh_;
    VertCoords clonePoints_;

    std::string name_;
};

/// Undo action for ObjectMesh topology only (not points) change
class ChangeMeshTopologyAction : public HistoryAction
{
public:
    using Obj = ObjectMesh;

    /// use this constructor to remember object's mesh points before making any changes in it
    ChangeMeshTopologyAction( std::string name, const std::shared_ptr<ObjectMesh>& obj ) :
        objMesh_{ obj },
        name_{ std::move( name ) }
    {
        if ( !objMesh_ )
            return;
        if ( auto m = objMesh_->mesh() )
            cloneTopology_ = m->topology;
    }

    /// use this constructor to remember object's mesh topology and immediate set new value
    ChangeMeshTopologyAction( std::string name, const std::shared_ptr<ObjectMesh>& obj, MeshTopology && newTopology ) :
        objMesh_{ obj },
        name_{ std::move( name ) }
    {
        cloneTopology_ = std::move( newTopology );
        action( HistoryAction::Type::Redo );
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !objMesh_ )
            return;

        if ( auto m = objMesh_->varMesh() )
        {
            std::swap( m->topology, cloneTopology_ );
            objMesh_->setDirtyFlags( DIRTY_FACE );
        }
    }

    static void setObjectDirty( const std::shared_ptr<ObjectMesh>& obj )
    {
        if ( obj )
            obj->setDirtyFlags( DIRTY_FACE );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + cloneTopology_.heapBytes();
    }

private:
    std::shared_ptr<ObjectMesh> objMesh_;
    MeshTopology cloneTopology_;

    std::string name_;
};

/// Undo action for ObjectMeshHolder texturePerFace change
class ChangeMeshTexturePerFaceAction : public HistoryAction
{
public:
    using Obj = ObjectMeshHolder;

    /// use this constructor to remember object's texturePerFace data before making any changes in them
    ChangeMeshTexturePerFaceAction( std::string name, const std::shared_ptr<ObjectMeshHolder>& obj ) :
        objMesh_{ obj },
        name_{ std::move( name ) }
    {
        if ( obj )
        {
            texturePerFace_ = obj->getTexturePerFace();
        }
    }

    /// use this constructor to remember object's texturePerFace data and immediate set new value
    ChangeMeshTexturePerFaceAction( std::string name, const std::shared_ptr<ObjectMeshHolder>& obj, Vector<TextureId, FaceId>&& newTexturePerFace ) :
        objMesh_{ obj },
        name_{ std::move( name ) }
    {
        if ( obj )
        {
            texturePerFace_ = std::move( newTexturePerFace );
            obj->updateTexturePerFace( texturePerFace_ );
        }
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !objMesh_ )
            return;

        objMesh_->updateTexturePerFace( texturePerFace_ );
    }

    static void setObjectDirty( const std::shared_ptr<ObjectMeshHolder>& obj )
    {
        if ( obj )
            obj->setDirtyFlags( DIRTY_TEXTURE_PER_FACE );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + texturePerFace_.heapBytes();
    }

private:
    Vector<TextureId, FaceId> texturePerFace_;
    std::shared_ptr<ObjectMeshHolder> objMesh_;
    std::string name_;
};

/// \}

} // namespace MR
