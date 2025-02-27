#pragma once
#include "MRMeshFwd.h"
#include "MRMeshOrPoints.h"
#include <variant>

namespace MR
{

class MeshOrPointsObjectHolder
{
public:
    MeshOrPointsObjectHolder( std::shared_ptr<ObjectMesh> om ) { set( om ); }
    MeshOrPointsObjectHolder( std::shared_ptr<ObjectPoints> op ) { set( op ); }

    void set( std::shared_ptr<ObjectMesh> om );
    MRMESH_API ObjectMesh* asObjectMesh() const;
    
    void set( std::shared_ptr<ObjectPoints> op );
    MRMESH_API ObjectPoints* asObjectPoints() const;


    void reset() { set( std::shared_ptr<ObjectMesh>{} ); }
    const std::shared_ptr<VisualObject>& operator->() const { return visualObject_; }
    MRMESH_API MeshOrPoints meshOrPoints() const;
private:
    std::variant<ObjectMesh*, ObjectPoints*> var_;
    std::shared_ptr<VisualObject> visualObject_;
};

}
