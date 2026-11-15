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
    std::vector<ObjectPtr> pending{ object };
    while ( !pending.empty() )
    {
        auto current = std::move( pending.back() );
        pending.pop_back();
        if ( supportedModel( current.get() ) )
            return true;
        for ( const auto &child : current->children() )
            pending.push_back( child );
    }
    return false;
}
Expected<void> rigidAncestry( const Object *object )
{
    for ( auto *current = object; current; current = current->parent() )
    {
        if ( !FeatureAlignment::isRigid( AffineXf3d( current->xf() ) ) )
            return unexpected( std::string( _t( "Physical parts and their ancestors must have rigid transforms. Scale, "
                                            "shear and mirror are unsupported." ) ) );
        auto views = current->xfsForAllViewports();
        if ( views.reset() )
            return unexpected( std::string( _t( "Per-viewport transforms are unsupported. Use the same part position in all viewports." ) ) );
    }
    return {};
}
} // namespace
bool below( const Object *object, const Object *ancestor )
{
    return object && ancestor && ( object == ancestor || object->isAncestor( ancestor ) );
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
    struct Frame
    {
        ObjectPtr object;
        size_t index{}, child{};
        bool geometry{};
        std::map<std::string, int> total, seen;
    };
    std::vector<Frame> pending;
    const auto push = [&]( const ObjectPtr &object, std::string path )
    {
        Frame frame;
        frame.object = object;
        frame.index = result.size();
        frame.geometry = supportedModel( object.get() );
        for ( const auto &child : object->children() )
            ++frame.total[child->name()];
        result.push_back( { object, std::move( path ), false } );
        pending.push_back( std::move( frame ) );
    };
    if ( root )
        push( root, {} );
    while ( !pending.empty() )
    {
        auto &frame = pending.back();
        const auto &children = frame.object->children();
        if ( frame.child < children.size() )
        {
            auto child = children[frame.child++];
            auto label = child->name();
            const auto ordinal = ++frame.seen[label];
            if ( frame.total[label] > 1 )
                label += " [" + std::to_string( ordinal ) + "]";
            const auto &parentPath = result[frame.index].path;
            push( child, parentPath.empty() ? label : parentPath + " / " + label );
        }
        else
        {
            const bool geometry = frame.geometry;
            result[frame.index].part = frame.object != root &&
                !std::dynamic_pointer_cast<FeatureObject>( frame.object ) && geometry;
            pending.pop_back();
            if ( !pending.empty() )
                pending.back().geometry = pending.back().geometry || geometry;
        }
    }
    return result;
}
Expected<OrderedRoles> hierarchyRoles( const Catalog &entries, const ObjectPtr &first, const ObjectPtr &second )
{
    const auto a =
        std::find_if( entries.begin(), entries.end(), [&]( const auto &e ) { return e.object == first && e.part; } );
    const auto b =
        std::find_if( entries.begin(), entries.end(), [&]( const auto &e ) { return e.object == second && e.part; } );
    if ( a == entries.end() || b == entries.end() || a == b )
        return unexpected( std::string( _t( "Choose two different mesh parts or assemblies from this scene." ) ) );
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
        return unexpected( std::string( _t( "Select exactly two parts, or features belonging to exactly two mesh parts. "
                                        "The higher part stays fixed." ) ) );
    return hierarchyRoles( entries, selected[0], selected[1] );
}
Expected<OrderedRoles> selectedHierarchyRoles( const ObjectPtr &root,
    const std::vector<std::shared_ptr<const Object>> &selection )
{
    const auto ancestors = [&]( const Object *object )
    {
        std::vector<const Object *> path;
        for ( auto *node = object; node; node = node->parent() )
        {
            path.push_back( node );
            if ( node == root.get() )
            {
                std::reverse( path.begin(), path.end() );
                return path;
            }
            const auto *parent = node->parent();
            if ( !parent || std::none_of( parent->constChildren().begin(), parent->constChildren().end(),
                    [&]( const auto &child ) { return child.get() == node; } ) )
                break;
        }
        return std::vector<const Object *>{};
    };
    std::vector<ObjectPtr> parts;
    const auto add = [&]( const ObjectPtr &object )
    {
        if ( std::find( parts.begin(), parts.end(), object ) == parts.end() )
            parts.push_back( object );
    };
    for ( const auto &object : selection )
        if ( object && object.get() != root.get() && !ancestors( object.get() ).empty() &&
             !std::dynamic_pointer_cast<const FeatureObject>( object ) &&
             hasPartGeometry( std::const_pointer_cast<Object>( object ) ) )
            add( std::const_pointer_cast<Object>( object ) );
    if ( parts.empty() )
        for ( const auto &object : selection )
            if ( object && std::dynamic_pointer_cast<const FeatureObject>( object ) &&
                 !ancestors( object.get() ).empty() )
                for ( auto *parent = object->parent(); parent; parent = parent->parent() )
                    if ( supportedModel( parent ) )
                    {
                        if ( parent != root.get() )
                            add( parent->getSharedPtr() );
                        break;
                    }
    if ( parts.size() != 2 )
        return unexpected( std::string( _t( "Select exactly two parts, or features belonging to exactly two mesh parts. "
                                        "The higher part stays fixed." ) ) );
    const auto first = ancestors( parts[0].get() ), second = ancestors( parts[1].get() );
    size_t common = 0;
    while ( common < first.size() && common < second.size() && first[common] == second[common] )
        ++common;
    if ( common == first.size() )
        return OrderedRoles{ parts[0], parts[1] };
    if ( common == second.size() )
        return OrderedRoles{ parts[1], parts[0] };
    for ( const auto &child : first[common - 1]->constChildren() )
    {
        if ( child.get() == first[common] )
            return OrderedRoles{ parts[0], parts[1] };
        if ( child.get() == second[common] )
            return OrderedRoles{ parts[1], parts[0] };
    }
    return unexpected( std::string( _t( "Choose two different mesh parts or assemblies from this scene." ) ) );
}
bool haveCompatiblePrimaryFeatures( const OrderedRoles &roles )
{
    const auto featureMask = [&]( const ObjectPtr &owner, bool fixed )
    {
        unsigned mask = 0;
        const auto visit = [&]( const auto &self, const ObjectPtr &object ) -> void
        {
            if ( fixed && object == roles.moving )
                return;
            if ( isPlane( object ) ) mask |= 1u;
            if ( isAxis( object ) ) mask |= 2u;
            if ( isCenter( object ) ) mask |= 4u;
            for ( const auto &child : object->children() )
                self( self, child );
        };
        if ( owner )
            visit( visit, owner );
        return mask;
    };
    return ( featureMask( roles.fixed, true ) & featureMask( roles.moving, false ) ) != 0;
}
Expected<Constraints> readConstraints( const ObjectPtr &root, const Selection &s, const Settings &settings )
{
    return readConstraints( root, catalog( root ), s, settings );
}
Expected<Constraints> readConstraints( const ObjectPtr &root, const Catalog &entries,
    const Selection &s, const Settings &settings )
{
    const auto present = [&]( const ObjectPtr &object )
    {
        return object &&
               std::any_of( entries.begin(), entries.end(), [&]( const auto &e ) { return e.object == object; } );
    };
    if ( !present( s.moving ) || !present( s.fixed ) || s.moving == root || s.fixed == root || s.moving == s.fixed )
        return unexpected( std::string( _t( "Choose different moving and fixed parts." ) ) );
    const auto isPart = [&]( const ObjectPtr &object )
    {
        return std::any_of( entries.begin(), entries.end(), [&]( const auto &e ) { return e.object == object && e.part; } );
    };
    if ( !isPart( s.moving ) || !isPart( s.fixed ) )
        return unexpected( std::string( _t( "Part roles must refer to meshes or assemblies, not fitted features." ) ) );
    auto ordered = hierarchyRoles( entries, s.fixed, s.moving );
    if ( !ordered || ( !s.explicitRoles && ordered->fixed != s.fixed ) )
        return unexpected(
            std::string( _t( "Scene-tree order changed. Reopen Feature Alignment or swap the assigned objects." ) ) );
    if ( below( s.fixed.get(), s.moving.get() ) )
        return unexpected(
            std::string( _t( "The moving assembly contains the fixed part. Choose a smaller moving branch." ) ) );
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
        return unexpected( std::string( _t( "Choose a primary feature belonging to each role. The fixed feature cannot "
                                        "belong to the moving branch." ) ) );
    FeatureAlignment::FeatureReferences references;
    references.primary = { std::dynamic_pointer_cast<const FeatureObject>( s.movingPrimary ),
                           std::dynamic_pointer_cast<const FeatureObject>( s.fixedPrimary ) };
    for ( int i = 0; i < 2; ++i )
    {
        if ( ( s.movingLocators[i] && !owner( s.movingLocators[i], true ) ) ||
             ( s.fixedLocators[i] && !owner( s.fixedLocators[i], false ) ) )
            return unexpected( std::string( _t( "A locator does not belong to its assigned part." ) ) );
        references.locators[i] = { std::dynamic_pointer_cast<const FeatureObject>( s.movingLocators[i] ),
                                   std::dynamic_pointer_cast<const FeatureObject>( s.fixedLocators[i] ) };
        if ( ( s.movingLocators[i] && !references.locators[i].moving ) ||
             ( s.fixedLocators[i] && !references.locators[i].fixed ) )
            return unexpected( std::string( _t( "Locators must be existing fitted features." ) ) );
    }
    if ( ( s.movingClock && !owner( s.movingClock, true ) ) || ( s.fixedClock && !owner( s.fixedClock, false ) ) )
        return unexpected( std::string( _t( "A clocking reference does not belong to its assigned part." ) ) );
    references.clocking = { std::dynamic_pointer_cast<const FeatureObject>( s.movingClock ),
                            std::dynamic_pointer_cast<const FeatureObject>( s.fixedClock ) };
    if ( ( s.movingClock && !references.clocking.moving ) || ( s.fixedClock && !references.clocking.fixed ) )
        return unexpected( std::string( _t( "Clocking references must be existing fitted features." ) ) );
    return FeatureAlignment::readConstraints( references, settings );
}
} // namespace MR::FeatureAlignmentTool
