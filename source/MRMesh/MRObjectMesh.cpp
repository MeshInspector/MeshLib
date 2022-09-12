#include "MRObjectMesh.h"
#include "MRObjectFactory.h"
#include "MRMesh.h"
#include "MRMeshComponents.h"
#include "MRMeshIntersect.h"
#include "MRStringConvert.h"
#include "MRAABBTree.h"
#include "MRLine3.h"
#include "MRGTest.h"
#include "MRPch/MRJson.h"
#include "MRPch/MRTBB.h"

namespace MR
{

MR_ADD_CLASS_FACTORY( ObjectMesh )


ObjectMesh::ObjectMesh( const ObjectMesh& other ) :
    ObjectMeshHolder( other )
{
}

std::optional<MeshIntersectionResult> ObjectMesh::worldRayIntersection( const Line3f& worldRay, const FaceBitSet* region ) const
{
    std::optional<MeshIntersectionResult> res;
    if ( !mesh_ )
        return res;
    const AffineXf3f rayToMeshXf = worldXf().inverse();
    res = rayMeshIntersect( { *mesh_, region }, transformed( worldRay, rayToMeshXf ) );
    return res;
}

void ObjectMesh::setMesh( std::shared_ptr< Mesh > mesh )
{
    if ( mesh == mesh_ )
        return;
    mesh_ = std::move(mesh);
    selectFaces({});
    selectEdges({});
    setCreases({});
    setDirtyFlags( DIRTY_ALL );
}

std::shared_ptr< Mesh > ObjectMesh::updateMesh( std::shared_ptr< Mesh > mesh )
{
    if ( mesh != mesh_ )
    {
        mesh_.swap( mesh );
        setDirtyFlags( DIRTY_ALL );
    }
    return mesh;
}

std::vector<std::string> ObjectMesh::getInfoLines() const
{
    std::vector<std::string> res = ObjectMeshHolder::getInfoLines();

    if ( mesh_ )
    {
        updateMeshStat_();

        res.push_back( "components: " + std::to_string( meshStat_->numComponents ) );
        res.push_back( "vertices: " + std::to_string( mesh_->topology.numValidVerts() ) );

        res.push_back( "faces: " + std::to_string( mesh_->topology.numValidFaces() ) );
        if( auto nFacesSelected = numSelectedFaces() )
            res.back() += " / " + std::to_string( nFacesSelected ) + " selected";

        res.push_back( "edges: " + std::to_string( meshStat_->numUndirectedEdges ) );
        if( auto nEdgesSelected = numSelectedEdges() )
            res.back() += " / " + std::to_string( nEdgesSelected ) + " selected";
        if( auto nCreaseEdges = numCreaseEdges() )
            res.back() += " / " + std::to_string( nCreaseEdges ) + " creases";

        res.push_back( "holes: " + std::to_string( meshStat_->numHoles ) );

        res.push_back( "area: " + std::to_string( totalArea() ) );

        boundingBoxToInfoLines_( res );

        if ( auto tree = mesh_->getAABBTreeNotCreate() )
            res.push_back( "AABB tree: " + bytesString( tree->heapBytes() ) );
    }
    else
        res.push_back( "no mesh" );
    return res;
}

std::shared_ptr<Object> ObjectMesh::clone() const
{
    auto res = std::make_shared<ObjectMesh>( ProtectedStruct{}, *this );
    if ( mesh_ )
        res->mesh_ = std::make_shared<Mesh>( *mesh_ );
    return res;
}

std::shared_ptr<Object> ObjectMesh::shallowClone() const
{
    auto res = std::make_shared<ObjectMesh>( ProtectedStruct{}, *this );
    if ( mesh_ )
        res->mesh_ = mesh_;
    return res;
}

void ObjectMesh::setDirtyFlags( uint32_t mask )
{
    ObjectMeshHolder::setDirtyFlags( mask );
    if ( mask & DIRTY_POSITION || mask & DIRTY_FACE)
    {
        if ( mesh_ )
        {
            meshChangedSignal( mask );
        }
    }
}

void ObjectMesh::swapBase_( Object& other )
{
    if ( auto otherMesh = other.asType<ObjectMesh>() )
        std::swap( *this, *otherMesh );
    else
        assert( false );
}

void ObjectMesh::swapSignals_( Object& other )
{
    ObjectMeshHolder::swapSignals_( other );
    if ( auto otherMesh = other.asType<ObjectMesh>() )
        std::swap( meshChangedSignal, otherMesh->meshChangedSignal );
    else
        assert( false );
}

void ObjectMesh::serializeFields_( Json::Value& root ) const
{
    ObjectMeshHolder::serializeFields_( root );
    root["Type"].append( ObjectMesh::TypeName() );
}

void ObjectMesh::Mirror( const Plane3f& plane )
{
    const auto xf = worldXf();
    const auto inverseXf = xf.inverse();
   
    for ( auto& p : mesh_->points )
    {
        auto pCopy = xf( p );
        pCopy += 2.0f * ( plane.project( pCopy ) - pCopy );
        p = inverseXf( pCopy );
    }

    mesh_->topology.flipOrientation();
    setDirtyFlags( DIRTY_FACE );    
}

TEST(MRMesh, DataModel)
{
    Object root;
    EXPECT_EQ(root.children().size(), 0);

    auto child = std::make_shared<Object>();
    EXPECT_TRUE(root.addChild(child));
    EXPECT_FALSE(root.addChild(child));
    EXPECT_EQ(&root, child->parent());
    EXPECT_EQ(root.children().size(), 1);

    child->setName( "child" );
    EXPECT_EQ( child, root.find( "child" ) );
    EXPECT_FALSE( root.find( "something" ) );
    EXPECT_EQ( child, root.find<Object>( "child" ) );
    EXPECT_FALSE( root.find<ObjectMesh>( "child" ) );

    auto grandchild = std::make_shared<ObjectMesh>();
    EXPECT_TRUE(child->addChild(grandchild));
    EXPECT_EQ(child.get(), grandchild->parent());

    EXPECT_TRUE(root.removeChild(child));
    EXPECT_FALSE(root.removeChild(child));
    EXPECT_EQ(nullptr, child->parent());
    EXPECT_EQ(root.children().size(), 0);

    child->removeAllChildren();
    EXPECT_EQ(child->children().size(), 0);
}

} //namespace MR
