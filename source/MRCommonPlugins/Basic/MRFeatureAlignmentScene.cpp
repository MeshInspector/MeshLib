#include "MRFeatureAlignmentScene.h"
#include <MRMesh/MRPlaneObject.h>
#include <MRMesh/MRCircleObject.h>
#include <MRMesh/MRCylinderObject.h>
#include <MRMesh/MRFeatureObject.h>
#include <MRMesh/MRObjectMesh.h>
#include <MRMesh/MRObjectPoints.h>
#include <MRMesh/MRObjectLines.h>
#include <MRMesh/MRPointObject.h>
#include <MRMesh/MRSphereObject.h>
#include <MRMesh/MRLineObject.h>
#include <algorithm>
#include <map>

namespace MR::FeatureAlignmentTool
{
namespace
{
bool supportedModel( const Object *object )
{
    return dynamic_cast<const ObjectMesh *>( object ) || dynamic_cast<const ObjectPoints *>( object ) ||
           dynamic_cast<const ObjectLines *>( object );
}
bool hasPartGeometry( const ObjectPtr &object )
{
    if ( supportedModel( object.get() ) )
        return true;
    for ( const auto &child : object->children() )
        if ( hasPartGeometry( child ) )
            return true;
    return false;
}
Expected<void> rigidAncestry( const Object *object )
{
    for ( auto *current = object; current; current = current->parent() )
    {
        if ( !FeatureAlignment::isRigid( AffineXf3d( current->xf() ) ) )
            return unexpected( std::string( "Physical parts and their ancestors must have rigid transforms. Scale, "
                                            "shear and mirror are unsupported." ) );
        auto views = current->xfsForAllViewports();
        if ( views.reset() )
            return unexpected( std::string(
                "Per-viewport transforms are unsupported. Use the same part position in all viewports." ) );
    }
    return {};
}
} // namespace
bool below( const Object *object, const Object *ancestor )
{
    if ( !ancestor )
        return false;
    for ( auto *current = object; current; current = current->parent() )
        if ( current == ancestor )
            return true;
    return false;
}
bool isHole( const ObjectPtr &object )
{
    return bool( std::dynamic_pointer_cast<CircleObject>( object ) ) ||
           bool( std::dynamic_pointer_cast<CylinderObject>( object ) );
}
bool isPlane( const ObjectPtr &object )
{
    return bool( std::dynamic_pointer_cast<PlaneObject>( object ) );
}
bool isAxis( const ObjectPtr &object )
{
    return isHole( object ) || bool( std::dynamic_pointer_cast<LineObject>( object ) );
}
bool isCenter( const ObjectPtr &object )
{
    return isHole( object ) || bool( std::dynamic_pointer_cast<PointObject>( object ) ) ||
           bool( std::dynamic_pointer_cast<SphereObject>( object ) );
}
bool isDirection( const ObjectPtr &object )
{
    return isPlane( object ) || isAxis( object );
}
Catalog catalog( const ObjectPtr &root )
{
    Catalog result;
    const auto visit = [&]( const auto &self, const ObjectPtr &object, const std::string &path ) -> void
    {
        result.push_back(
            { object, path,
              object != root && !std::dynamic_pointer_cast<FeatureObject>( object ) && hasPartGeometry( object ) } );
        std::map<std::string, int> total, seen;
        for ( const auto &child : object->children() )
            ++total[child->name()];
        for ( const auto &child : object->children() )
        {
            auto label = child->name();
            const auto ordinal = ++seen[label];
            if ( total[label] > 1 )
                label += " [" + std::to_string( ordinal ) + "]";
            self( self, child, path.empty() ? label : path + " / " + label );
        }
    };
    if ( root )
        visit( visit, root, {} );
    return result;
}
Expected<ObjectPtr> oneSelectedPart( const Catalog &entries )
{
    ObjectPtr selected;
    for ( const auto &e : entries )
        if ( e.part && e.object->isSelected() )
        {
            if ( selected )
                return unexpected( std::string( "Select exactly one part for this role. Click order is not used." ) );
            selected = e.object;
        }
    if ( !selected )
        return unexpected( std::string( "Select one mesh or assembly in the scene tree, then choose its role." ) );
    return selected;
}
Expected<OrderedRoles> hierarchyRoles( const Catalog &entries, const ObjectPtr &first, const ObjectPtr &second )
{
    const auto a =
        std::find_if( entries.begin(), entries.end(), [&]( const auto &e ) { return e.object == first && e.part; } );
    const auto b =
        std::find_if( entries.begin(), entries.end(), [&]( const auto &e ) { return e.object == second && e.part; } );
    if ( a == entries.end() || b == entries.end() || a == b )
        return unexpected( std::string( "Choose two different mesh parts or assemblies from this scene." ) );
    return a < b ? OrderedRoles{ first, second } : OrderedRoles{ second, first };
}
Expected<OrderedRoles> selectedHierarchyRoles( const Catalog &entries )
{
    std::vector<ObjectPtr> selected;
    for ( const auto &e : entries )
        if ( e.part && e.object->isSelected() )
            selected.push_back( e.object );
    if ( selected.empty() )
    {
        // Existing selected features identify their closest owning mesh. Their
        // selection/click order never chooses the fixed or moving role.
        for ( const auto &e : entries )
            if ( std::dynamic_pointer_cast<FeatureObject>( e.object ) && e.object->isSelected() )
            {
                for ( auto *parent = e.object->parent(); parent; parent = parent->parent() )
                    if ( supportedModel( parent ) )
                    {
                        const auto found = std::find_if( entries.begin(), entries.end(), [&]( const auto &item )
                                                         { return item.object.get() == parent; } );
                        if ( found != entries.end() &&
                             std::find( selected.begin(), selected.end(), found->object ) == selected.end() )
                            selected.push_back( found->object );
                        break;
                    }
            }
    }
    if ( selected.size() != 2 )
        return unexpected( std::string( "Select exactly two parts, or features belonging to exactly two mesh parts. "
                                        "The higher part stays fixed." ) );
    return hierarchyRoles( entries, selected[0], selected[1] );
}
Expected<Constraints> readConstraints( const ObjectPtr &root, const Selection &s, const Settings &settings )
{
    const auto entries = catalog( root );
    const auto present = [&]( const ObjectPtr &object )
    {
        return object &&
               std::any_of( entries.begin(), entries.end(), [&]( const auto &e ) { return e.object == object; } );
    };
    if ( !present( s.moving ) || !present( s.fixed ) || s.moving == root || s.fixed == root || s.moving == s.fixed )
        return unexpected( std::string( "Choose different moving and fixed parts." ) );
    if ( !hasPartGeometry( s.moving ) || !hasPartGeometry( s.fixed ) ||
         std::dynamic_pointer_cast<FeatureObject>( s.moving ) || std::dynamic_pointer_cast<FeatureObject>( s.fixed ) )
        return unexpected( std::string( "Part roles must refer to meshes or assemblies, not fitted features." ) );
    auto ordered = hierarchyRoles( entries, s.fixed, s.moving );
    if ( !ordered || ordered->fixed != s.fixed )
        return unexpected(
            std::string( "The higher part in the scene tree must stay fixed; only the lower part can move." ) );
    if ( below( s.fixed.get(), s.moving.get() ) )
        return unexpected(
            std::string( "The moving assembly contains the fixed part. Choose a smaller moving branch." ) );
    for ( const auto &part : { s.moving, s.fixed } )
    {
        auto rigid = rigidAncestry( part.get() );
        if ( !rigid )
            return unexpected( rigid.error() );
    }
    const auto owner = [&]( const ObjectPtr &feature, bool moving ) -> bool
    {
        if ( !present( feature ) )
            return false;
        return below( feature.get(), ( moving ? s.moving : s.fixed ).get() ) &&
               ( moving || !below( feature.get(), s.moving.get() ) );
    };
    if ( !owner( s.movingPrimary, true ) || !owner( s.fixedPrimary, false ) )
        return unexpected( std::string( "Choose a primary feature belonging to each role. The fixed feature cannot "
                                        "belong to the moving branch." ) );
    FeatureAlignment::FeatureReferences references;
    references.primary = { std::dynamic_pointer_cast<const FeatureObject>( s.movingPrimary ),
                           std::dynamic_pointer_cast<const FeatureObject>( s.fixedPrimary ) };
    for ( int i = 0; i < 2; ++i )
    {
        if ( bool( s.movingLocators[i] ) != bool( s.fixedLocators[i] ) )
            return unexpected( std::string( "Choose both sides of locator " ) + std::to_string( i + 1 ) +
                               " or leave both empty." );
        if ( !s.movingLocators[i] )
            continue;
        if ( i == 1 && ( !s.movingLocators[0] || s.movingLocators[1] == s.movingLocators[0] ||
                         s.fixedLocators[1] == s.fixedLocators[0] ) )
            return unexpected( std::string( "The second locator must be a different pair after the first locator." ) );
        if ( !owner( s.movingLocators[i], true ) || !owner( s.fixedLocators[i], false ) )
            return unexpected( std::string( "A locator does not belong to its assigned part." ) );
        references.locators[i] = { std::dynamic_pointer_cast<const FeatureObject>( s.movingLocators[i] ),
                                   std::dynamic_pointer_cast<const FeatureObject>( s.fixedLocators[i] ) };
        if ( !references.locators[i].moving || !references.locators[i].fixed )
            return unexpected( std::string( "Locators must be existing fitted features." ) );
    }
    if ( bool( s.movingClock ) != bool( s.fixedClock ) )
        return unexpected( std::string( "Choose both clocking references or leave both empty." ) );
    if ( s.movingClock )
    {
        if ( !owner( s.movingClock, true ) || !owner( s.fixedClock, false ) )
            return unexpected( std::string( "A clocking reference does not belong to its assigned part." ) );
        references.clocking = { std::dynamic_pointer_cast<const FeatureObject>( s.movingClock ),
                                std::dynamic_pointer_cast<const FeatureObject>( s.fixedClock ) };
        if ( !references.clocking.moving || !references.clocking.fixed )
            return unexpected( std::string( "Clocking references must be existing fitted features." ) );
    }
    return FeatureAlignment::readConstraints( references, settings );
}

Expected<SceneStamp> stamp( const ObjectPtr &root )
{
    SceneStamp result;
    for ( const auto &e : catalog( root ) )
    {
        auto views = e.object->xfsForAllViewports();
        if ( views.reset() )
            return unexpected( std::string( "Per-viewport transforms are unsupported." ) );
        const void *token = nullptr;
        if ( const auto mesh = std::dynamic_pointer_cast<ObjectMesh>( e.object ) )
            token = mesh->meshPtr();
        else if ( const auto points = std::dynamic_pointer_cast<ObjectPoints>( e.object ) )
            token = points->pointCloudPtr();
        else if ( const auto lines = std::dynamic_pointer_cast<ObjectLines>( e.object ) )
            token = lines->polylinePtr();
        result.nodes.push_back( { e.object, e.object->parent(), e.object->xf(), token, e.object->name() } );
    }
    return result;
}
Expected<bool> SceneStamp::matches( const ObjectPtr &root ) const
{
    auto current = stamp( root );
    if ( !current )
        return unexpected( current.error() );
    if ( current->nodes.size() != nodes.size() )
        return false;
    for ( size_t i = 0; i < nodes.size(); ++i )
    {
        const auto &a = nodes[i];
        const auto &b = current->nodes[i];
        if ( a.object != b.object || a.parent != b.parent || a.local != b.local || a.modelToken != b.modelToken ||
             a.name != b.name )
            return false;
    }
    return true;
}
} // namespace MR::FeatureAlignmentTool
