#pragma once
#include "MRMeshFwd.h"
#include "MRMeshOrPoints.h"
#include "MRVisualObject.h"
#include <variant>

namespace MR
{

class MeshOrPointsObjectHolder
{
public:
    MeshOrPointsObjectHolder() { reset(); }
    MRMESH_API MeshOrPointsObjectHolder( std::shared_ptr<VisualObject> vo );
    MeshOrPointsObjectHolder( std::shared_ptr<ObjectMesh> om ) { set( om ); }
    MeshOrPointsObjectHolder( std::shared_ptr<ObjectPoints> op ) { set( op ); }

    MRMESH_API void set( std::shared_ptr<ObjectMesh> om );
    MRMESH_API ObjectMesh* asObjectMesh() const;
    
    MRMESH_API void set( std::shared_ptr<ObjectPoints> op );
    MRMESH_API ObjectPoints* asObjectPoints() const;


    void reset() { set( std::shared_ptr<ObjectMesh>{} ); }
    const std::shared_ptr<VisualObject>& operator->() const { return visualObject_; }
    const std::shared_ptr<VisualObject>& get() const { return visualObject_; }
    bool operator==( std::shared_ptr<VisualObject> other ) const { return visualObject_ == other; }


    MRMESH_API MeshOrPoints meshOrPoints() const;
private:
    std::variant<ObjectMesh*, ObjectPoints*> var_ = ( ObjectMesh* ) nullptr;
    std::shared_ptr<VisualObject> visualObject_ = {};
};

}
