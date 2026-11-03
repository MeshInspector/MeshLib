#include "MRPythonScene.h"
#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRObjectLinesHolder.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRObjectDistanceMap.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRPch/MRFmt.h"

#ifndef MESHLIB_NO_VOXELS
#include "MRVoxels/MRObjectVoxels.h"
#endif

namespace MR::PythonViewer
{

namespace
{

// The module releases the GIL around the calls, so these can wait for the GUI thread.
template <typename F>
void runInGuiThread( F&& f )
{
    CommandLoop::runCommandFromGUIThread( std::forward<F>( f ) );
}

template <typename T>
void selectType()
{
    auto selected = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Any );
    for ( auto& obj : selected )
        obj->select( bool( obj->asType<T>() ) );
}

void unselect()
{
    auto selected = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    for ( auto& obj : selected )
        obj->select( false );
}

template <typename ObjectType, typename ModelType, typename Setter>
void addModelToScene( const ModelType& model, const std::string& name, Setter&& setter )
{
    runInGuiThread( [&] ()
    {
        auto newObject = std::make_shared<ObjectType>();
        setter( *newObject, std::make_shared<ModelType>( model ) );
        newObject->setName( name );
        SceneRoot::get().addChild( newObject );
    } );
}

// The models of the selected objects of type `ObjectType`, copied (`getModel` returns a pointer or a value).
template <typename ObjectType, typename ModelType, typename GetModel>
std::vector<ModelType> getSelectedModels( GetModel&& getModel )
{
    std::vector<ModelType> ret;
    runInGuiThread( [&]
    {
        auto objects = getAllObjectsInTree<ObjectType>( SceneRoot::get(), ObjectSelectivityType::Selected );
        ret.reserve( objects.size() );
        for ( const auto& object : objects )
        {
            if constexpr ( std::is_same_v<std::remove_cvref_t<decltype( getModel( *object ) )>, ModelType> )
                ret.push_back( getModel( *object ) );
            else
                ret.push_back( *getModel( *object ) );
        }
    } );
    return ret;
}

template <typename ObjectType, typename BitSetType, typename GetBitSet>
std::vector<BitSetType> getSelectedBitsets( GetBitSet&& getBitSet )
{
    std::vector<BitSetType> ret;
    runInGuiThread( [&]
    {
        auto selected = getAllObjectsInTree<ObjectType>( &SceneRoot::get(), ObjectSelectivityType::Selected );
        ret.resize( selected.size() );
        for ( std::size_t i = 0; i < ret.size(); i++ )
            ret[i] = getBitSet( *selected[i] );
    } );
    return ret;
}

template <typename ObjectType, typename BitSetType, typename SetBitSet>
void setSelectedBitsets( const std::vector<BitSetType>& bitsets, SetBitSet&& setBitSet )
{
    runInGuiThread( [&]
    {
        auto selected = getAllObjectsInTree<ObjectType>( &SceneRoot::get(), ObjectSelectivityType::Selected );
        if ( selected.size() != bitsets.size() )
            throw std::runtime_error( fmt::format( "Specified {} bitsets, but {} objects are selected.", bitsets.size(), selected.size() ) );
        for ( std::size_t i = 0; i < selected.size(); i++ )
            setBitSet( *selected[i], bitsets[i] );
    } );
}

} // namespace

void addMeshToScene( const Mesh& mesh, const std::string& name )
{
    addModelToScene<ObjectMesh>( mesh, name, [] ( ObjectMesh& o, std::shared_ptr<Mesh> m ) { o.setMesh( std::move( m ) ); } );
}

void addPointCloudToScene( const PointCloud& points, const std::string& name )
{
    addModelToScene<ObjectPoints>( points, name, [] ( ObjectPoints& o, std::shared_ptr<PointCloud> p ) { o.setPointCloud( std::move( p ) ); } );
}

void addLinesToScene( const Polyline3& lines, const std::string& name )
{
    addModelToScene<ObjectLines>( lines, name, [] ( ObjectLines& o, std::shared_ptr<Polyline3> l ) { o.setPolyline( std::move( l ) ); } );
}

void addDistanceMapToScene( const DistanceMap& distancemap, const std::string& name, const AffineXf3f& dmap_to_local_xf )
{
    addModelToScene<ObjectDistanceMap>( distancemap, name, [&] ( ObjectDistanceMap& o, std::shared_ptr<DistanceMap> d ) { o.setDistanceMap( std::move( d ), dmap_to_local_xf ); } );
}

void modifySelectedMesh( const Mesh& mesh )
{
    runInGuiThread( [&]
    {
        auto selected = getAllObjectsInTree<ObjectMesh>( &SceneRoot::get(), ObjectSelectivityType::Selected );
        if ( selected.size() != 1 )
            throw std::runtime_error( fmt::format( "Exactly one mesh must be selected, but have {}.", selected.size() ) );
        if ( !selected[0] )
            throw std::runtime_error( "Internal error (the object is null?)." );
        selected[0]->setMesh( std::make_shared<Mesh>( mesh ) );
        selected[0]->setDirtyFlags( DIRTY_ALL );
    } );
}

std::vector<FaceBitSet> getSelectedMeshFaces()
{
    return getSelectedBitsets<ObjectMeshHolder, FaceBitSet>( [] ( const ObjectMeshHolder& o ) { return o.getSelectedFaces(); } );
}

std::vector<UndirectedEdgeBitSet> getSelectedMeshEdges()
{
    return getSelectedBitsets<ObjectMeshHolder, UndirectedEdgeBitSet>( [] ( const ObjectMeshHolder& o ) { return o.getSelectedEdges(); } );
}

std::vector<VertBitSet> getSelectedPointCloudPoints()
{
    return getSelectedBitsets<ObjectPointsHolder, VertBitSet>( [] ( const ObjectPointsHolder& o ) { return o.getSelectedPoints(); } );
}

void setSelectedMeshFaces( const std::vector<FaceBitSet>& bitsets )
{
    setSelectedBitsets<ObjectMeshHolder>( bitsets, [] ( ObjectMeshHolder& o, const FaceBitSet& b ) { o.selectFaces( b ); } );
}

void setSelectedMeshEdges( const std::vector<UndirectedEdgeBitSet>& bitsets )
{
    setSelectedBitsets<ObjectMeshHolder>( bitsets, [] ( ObjectMeshHolder& o, const UndirectedEdgeBitSet& b ) { o.selectEdges( b ); } );
}

void setSelectedPointCloudPoints( const std::vector<VertBitSet>& bitsets )
{
    setSelectedBitsets<ObjectPointsHolder>( bitsets, [] ( ObjectPointsHolder& o, const VertBitSet& b ) { o.selectPoints( b ); } );
}

void clearScene()
{
    runInGuiThread( [] { SceneRoot::get().removeAllChildren(); } );
}

void selectByName( const std::string& objectName )
{
    runInGuiThread( [&]
    {
        auto selected = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Any );
        for ( auto& obj : selected )
            obj->select( objectName == obj->name() );
    } );
}

void selectByType( const std::string& typeName )
{
    runInGuiThread( [&]
    {
        if ( typeName == "Meshes" )
            return selectType<ObjectMesh>();
        if ( typeName == "Points" )
            return selectType<ObjectPoints>();
#ifndef MESHLIB_NO_VOXELS
        if ( typeName == "Voxels" )
            return selectType<ObjectVoxels>();
#endif
        unselect();
    } );
}

void unselectAll()
{
    runInGuiThread( [] { unselect(); } );
}

std::vector<std::shared_ptr<Object>> getSelectedObjects()
{
    return getAllObjectsInTree( &SceneRoot::get(), ObjectSelectivityType::Selected );
}

std::vector<Mesh> getSelectedMeshes()
{
    return getSelectedModels<ObjectMeshHolder, Mesh>( [] ( const ObjectMeshHolder& o ) { return o.meshPtr(); } );
}

std::vector<PointCloud> getSelectedPointClouds()
{
    return getSelectedModels<ObjectPointsHolder, PointCloud>( [] ( const ObjectPointsHolder& o ) { return o.pointCloudPtr(); } );
}

std::vector<Polyline3> getSelectedPolylines()
{
    return getSelectedModels<ObjectLinesHolder, Polyline3>( [] ( const ObjectLinesHolder& o ) { return o.polylinePtr(); } );
}

std::vector<DistanceMap> getSelectedDistanceMaps()
{
    return getSelectedModels<ObjectDistanceMap, DistanceMap>( [] ( const ObjectDistanceMap& o ) { return o.getDistanceMap(); } );
}

#ifndef MESHLIB_NO_VOXELS
void addVoxelsToScene( const VdbVolume& voxels, const std::string& name )
{
    runInGuiThread( [&] ()
    {
        auto newObject = std::make_shared<ObjectVoxels>();
        newObject->construct( voxels );
        auto bins = newObject->histogram().getBins();
        auto minMax = newObject->histogram().getBinMinMax( bins.size() / 3 );
        (void)newObject->setIsoValue( minMax.first ); //TODO: process potential error
        newObject->setName( name );
        SceneRoot::get().addChild( newObject );
    } );
}

std::vector<VdbVolume> getSelectedVoxels()
{
    return getSelectedModels<ObjectVoxels, VdbVolume>( [] ( const ObjectVoxels& o ) { return o.vdbVolume(); } );
}
#endif

} // namespace MR::PythonViewer
