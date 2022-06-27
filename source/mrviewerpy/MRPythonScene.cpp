#include "MRMesh/MRPython.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRObjectVoxels.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRViewer/MRCommandLoop.h"

namespace MR
{

bool selectName( const std::string& modelName )
{
    auto selected = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Any );
    for ( auto& obj : selected )
    {
        if ( modelName == obj->name() )
        {
            obj->select( true );
        }
        else
        {
            obj->select( false );
        }
    }
    return true;
}

template <typename T>
bool selectType()
{
    auto selected = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Any );
    for ( auto& obj : selected )
    {
        if ( auto objConverted = obj->asType<T>() )
        {
            obj->select( true );
        }
        else
        {
            obj->select( false );
        }
    }
    return true;
}

bool unselect()
{
    auto selected = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    for ( auto& obj : selected )
    {
        obj->select( false );
    }
    return true;
}

} //namespace MR


void pythonSelectName( const std::string modelName )
{
    MR::CommandLoop::runCommandFromGUIThread( [modelName] ()
    {
        MR::selectName( modelName );
    } );
}
void pythonUnselect()
{
    MR::CommandLoop::runCommandFromGUIThread( [&] ()
    {
        MR::unselect();
    } );
}
void pythonSelectType( const std::string modelType )
{
    MR::CommandLoop::runCommandFromGUIThread( [modelType] ()
    {
        if ( modelType == "Meshes" )
        {
            MR::selectType<MR::ObjectMesh>();
            return;
        }
        if ( modelType == "Points" )
        {
            MR::selectType<MR::ObjectPoints>();
            return;
        }
        if ( modelType == "Voxels" )
        {
            MR::selectType<MR::ObjectVoxels>();
            return;
        }
        MR::unselect();
    } );
}

void pythonClearScene()
{
    MR::CommandLoop::runCommandFromGUIThread( [] ()
    {
        MR::SceneRoot::get().removeAllChildren();
    } );
}

void pythonAddMeshToScene( const MR::Mesh& mesh, const std::string& name )
{
    MR::CommandLoop::runCommandFromGUIThread( [&] ()
    {
        std::shared_ptr<MR::ObjectMesh> objMesh = std::make_shared<MR::ObjectMesh>();
        objMesh->setMesh( std::make_shared<MR::Mesh>( mesh ) );
        objMesh->setName( name );
        MR::SceneRoot::get().addChild( objMesh );
    } );
}

void pythonAddPointCloudToScene( const MR::PointCloud& points, const std::string& name )
{
    MR::CommandLoop::runCommandFromGUIThread( [&] ()
    {
        std::shared_ptr<MR::ObjectPoints> objPoints = std::make_shared<MR::ObjectPoints>();
        objPoints->setPointCloud( std::make_shared<MR::PointCloud>( points ) );
        objPoints->setName( name );
        MR::SceneRoot::get().addChild( objPoints );
    } );
}

MR_ADD_PYTHON_CUSTOM_DEF( mrviewerpy, Scene, [] ( pybind11::module_& m )
{
    m.def( "add_mesh_to_scene", &pythonAddMeshToScene, "add mesh to scene" );
    m.def( "add_point_cloud_to_scene", &pythonAddPointCloudToScene, "add point cloud to scene" );
    m.def( "clear_scene", &pythonClearScene, "remove all models" );
    m.def( "select_by_name", &pythonSelectName, "select only models with that name, unselect others" );
    m.def( "select_by_type", &pythonSelectType, "string typeName: {\"Meshes\", \"Points\", \"Voxels\"}\nselect only models with that name, unselect others" );
    m.def( "unselect_all", &pythonUnselect, "unselect all models" );
} )
