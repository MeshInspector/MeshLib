#pragma once
#include "MRMeshFwd.h"
#include <variant>

namespace MR
{

/// This class can hold either ObjectMesh or ObjectPoint
/// It is used for convenient storage and operation with any of them
class MeshOrPointsObject
{
public:
    MeshOrPointsObject() { reset(); }
    /// construct, automatically detecting the object type (ObjectMesh or ObjectPoint)
    /// if set an another type, will be reset
    MRMESH_API MeshOrPointsObject( std::shared_ptr<VisualObject> vo );
    MeshOrPointsObject( std::shared_ptr<ObjectMesh> om ) { set( std::move( om ) ); }
    MeshOrPointsObject( std::shared_ptr<ObjectPoints> op ) { set( std::move( op ) ); }

    /// set to hold ObjectMesh
    MRMESH_API void set( std::shared_ptr<ObjectMesh> om );
    /// if holding ObjectMesh, return pointer to it, otherwise return nullptr
    MRMESH_API ObjectMesh* asObjectMesh() const;
    
    /// set to hold ObjectPoints
    MRMESH_API void set( std::shared_ptr<ObjectPoints> op );
    /// if holding ObjectPoints, return pointer to it, otherwise return nullptr
    MRMESH_API ObjectPoints* asObjectPoints() const;

    void reset() { set( std::shared_ptr<ObjectMesh>{} ); }
    const std::shared_ptr<VisualObject>& operator->() const { return visualObject_; }
    const std::shared_ptr<VisualObject>& get() const { return visualObject_; }
    bool operator==( std::shared_ptr<VisualObject> other ) const { return visualObject_ == other; }

    /// get class that hold either mesh part or point cloud
    MRMESH_API MeshOrPoints meshOrPoints() const;
private:
    std::variant<ObjectMesh*, ObjectPoints*> var_;
    std::shared_ptr<VisualObject> visualObject_;
};

}
