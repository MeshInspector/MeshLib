#pragma once
#include "exports.h"
#include "MRMesh/MRObject.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRObjectVoxels.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRObjectDistanceMap.h"
#include <memory>
#include <vector>
#include <string>

namespace MR
{

class Object;

// Interface for checking scene state, to determine availability, also can return string with requirements 
class ISceneStateCheck
{
public:
    virtual ~ISceneStateCheck() = default;
    // return empty string if all requirements are satisfied, otherwise return first unsatisfied requirement
    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const { return {}; }
    // return not-empty string with tooltip that shall replace the static tooltip from json
    virtual std::string getDynamicTooltip() const { return {}; }
};

// special namespace not to have signature conflicts
namespace ModelCheck
{
inline bool model( const Object& )
{
    return true;
}
inline bool model( const ObjectMesh& obj )
{
    return bool( obj.mesh() );
}
#ifndef __EMSCRIPTEN__
inline bool model( const ObjectVoxels& obj )
{
    return bool( obj.grid() );
}
#endif
inline bool model( const ObjectPoints& obj )
{
    return bool( obj.pointCloud() );
}
inline bool model( const ObjectLines& obj )
{
    return bool( obj.polyline() );
}
inline bool model( const ObjectDistanceMap& obj )
{
    return bool( obj.getDistanceMap() );
}
}

inline bool hasVisualRepresentation( const Object& )
{
    return true;
}

inline bool hasVisualRepresentation( const VisualObject& obj )
{
    return obj.hasVisualRepresentation();
}

// special struct for disabling visual representation check
struct NoVisualRepresentationCheck {};

// check that given vector has exactly N objects if type ObjectT
// returns error message if requirements are not satisfied
template<typename ObjectT, bool visualRepresentationCheck>
std::string sceneSelectedExactly( const std::vector<std::shared_ptr<const Object>>& objs, unsigned n )
{
    if ( objs.size() != n )
        return "Exactly " + std::to_string( n ) + " " + ObjectT::TypeName() + "(s) must be selected";
    for ( const auto& obj : objs )
    {
        auto tObj = obj->asType<ObjectT>();
        if ( !tObj )
            return std::string( "Selected object(s) must have type: " ) + ObjectT::TypeName();

        if ( !ModelCheck::model( *tObj ) )
            return "Selected object(s) must have valid model";

        if constexpr ( visualRepresentationCheck )
            if ( !hasVisualRepresentation( *tObj ) )
                return "Selected object(s) must have valid visual representation";
    }
    return "";
}

// checks that given vector has at least N objects if type ObjectT
// returns error message if requirements are not satisfied
template<typename ObjectT, bool visualRepresentationCheck>
std::string sceneSelectedAtLeast( const std::vector<std::shared_ptr<const Object>>& objs, unsigned n )
{
    if ( objs.size() < n )
        return "At least " + std::to_string( n ) + " " + ObjectT::TypeName() + "(s) must be selected";
    unsigned i = 0;
    for ( const auto& obj : objs )
    {
        auto tObj = obj->asType<ObjectT>();
        if ( !tObj )
            continue;
        if ( !ModelCheck::model( *tObj ) )
            continue;
        if constexpr ( visualRepresentationCheck )
            if ( !hasVisualRepresentation( *tObj ) )
                continue;
        ++i;
    }
    return ( i >= n ) ? 
        "" : 
        ( "At least " + std::to_string( n ) + " selected object(s) must have type: " + ObjectT::TypeName() + " with valid model" );
}

// check that given vector has exactly N objects if type ObjectT
template<unsigned N, typename ObjectT, typename = void>
class SceneStateExactCheck : virtual public ISceneStateCheck
{
public:
    virtual ~SceneStateExactCheck() = default;
    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& objs ) const override
    {
        return sceneSelectedExactly<ObjectT, true>(objs, N);
    }
};

template<unsigned N, typename ObjectT>
class SceneStateExactCheck<N, ObjectT, NoVisualRepresentationCheck> : virtual public ISceneStateCheck
{
public:
    virtual ~SceneStateExactCheck() = default;
    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& objs ) const override
    {
        return sceneSelectedExactly<ObjectT, false>(objs, N);
    }
};

// checks that given vector has at least N objects if type ObjectT
template<unsigned N, typename ObjectT, typename = void>
class SceneStateAtLeastCheck : virtual public ISceneStateCheck
{
public:
    virtual ~SceneStateAtLeastCheck() = default;
    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& objs ) const override
    {
        return sceneSelectedAtLeast<ObjectT, true>( objs, N );
    }
};

template<unsigned N, typename ObjectT>
class SceneStateAtLeastCheck<N, ObjectT, NoVisualRepresentationCheck> : virtual public ISceneStateCheck
{
public:
    virtual ~SceneStateAtLeastCheck() = default;
    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& objs ) const override
    {
        return sceneSelectedAtLeast<ObjectT, false>( objs, N );
    }
};

// checks that at least one of argument checks is true
template<typename ...Checks>
class SceneStateOrCheck : virtual public Checks...
{
public:
    virtual ~SceneStateOrCheck() = default;
    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>&objs ) const override
    {
        std::vector<std::string> checkRes;
        checkRes.reserve( sizeof...( Checks ) );
        ( checkRes.push_back( Checks::isAvailable( objs ) ), ... );
        std::string combinedRes;
        for ( int i = 0; i + 1 < checkRes.size(); ++i )
        {
            if ( checkRes[i].empty() )
                return "";
            combinedRes += checkRes[i];
            combinedRes += " or ";
        }
        if ( checkRes.back().empty() )
            return "";
        combinedRes += checkRes.back();
        return combinedRes;
    }
};

// checks that all of argument checks are true
template<typename ...Checks>
class SceneStateAndCheck : virtual public Checks...
{
public:
    virtual ~SceneStateAndCheck() = default;
    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& objs ) const override
    {
        std::vector<std::string> checkRes;
        checkRes.reserve( sizeof...( Checks ) );
        ( checkRes.push_back( Checks::isAvailable( objs ) ), ... );
        for ( const auto& res : checkRes )
        {
            if ( !res.empty() )
                return res;
        }
        return "";
    }
};


}