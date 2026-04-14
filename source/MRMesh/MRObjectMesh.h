#pragma once

#include "MRObjectMeshHolder.h"

namespace MR
{

/// an object that stores a mesh
/// \ingroup DataModelGroup
class MRMESH_CLASS ObjectMesh : public ObjectMeshHolder
{
public:
    ObjectMesh() = default;

    ObjectMesh( ObjectMesh&& ) noexcept = default;
    ObjectMesh& operator = ( ObjectMesh&& ) noexcept = default;

    constexpr static const char* StaticTypeName() noexcept { return "ObjectMesh"; }
    virtual const char* typeName() const override { return StaticTypeName(); }

    constexpr static const char* StaticClassName() noexcept { return "Mesh"; }
    virtual std::string className() const override { return StaticClassName(); }

    constexpr static const char* StaticClassNameInPlural() noexcept { return "Meshes"; }
    virtual std::string classNameInPlural() const override { return StaticClassNameInPlural(); }

    /// returns variable mesh, if const mesh is needed use `mesh()` instead
    virtual const std::shared_ptr< Mesh > & varMesh() { return data_.mesh; }

    /// sets given mesh to this, resets selection and creases
    MRMESH_API virtual void setMesh( std::shared_ptr< Mesh > mesh );
    /// sets given mesh to this, and returns back previous mesh of this;
    /// does not touch selection or creases
    MRMESH_API virtual std::shared_ptr< Mesh > updateMesh( std::shared_ptr< Mesh > mesh );

    MRMESH_API virtual std::vector<std::string> getInfoLines() const override;

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    /// \note this ctor is public only for std::make_shared used inside clone()
    ObjectMesh( ProtectedStruct, const ObjectMesh& obj ) : ObjectMesh( obj ) {}

    /// given ray in world coordinates, e.g. obtained from Viewport::unprojectPixelRay;
    /// finds its intersection with the mesh of this object considering its transformation relative to the world;
    /// it is inefficient to call this function for many rays, because it computes world-to-local xf every time
    MRMESH_API MeshIntersectionResult worldRayIntersection( const Line3f& worldRay, const FaceBitSet* region = nullptr ) const;

protected:
    ObjectMesh( const ObjectMesh& other ) = default;

    /// swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;
};

/// options to better control MR::merge function
struct ObjectMeshMergeOptions
{
    /// if not nullptr: merged object will take overridden transform for each object
    const std::vector<AffineXf3f>* overrideXfs{ nullptr };
};

/// constructs new ObjectMesh containing the union of valid data from all input objects
[[nodiscard]] MRMESH_API std::shared_ptr<ObjectMesh> merge( const std::vector<std::shared_ptr<ObjectMesh>>& objsMesh, const ObjectMeshMergeOptions& options = {} );

/// constructs new ObjectMesh containing the region of data from input object
/// does not copy selection
[[nodiscard]] MRMESH_API std::shared_ptr<ObjectMesh> cloneRegion( const std::shared_ptr<ObjectMesh>& objMesh, const FaceBitSet& region, bool copyTexture = true );

} ///namespace MR
