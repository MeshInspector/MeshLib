#pragma once

#include "MRISceneStateCheck.h"
#include "MRMesh/MRObject.h"

namespace MR
{

// special struct for disabling visual representation check
struct NoVisualRepresentationCheck {};

// special struct for disabling model check
struct NoModelCheck {};

// check that given vector has exactly N objects if type ObjectT
// returns error message if requirements are not satisfied
template<typename ObjectT, bool visualRepresentationCheck, bool modelCheck>
std::string sceneSelectedExactly( const std::vector<std::shared_ptr<const Object>>& objs, unsigned n )
{
    if ( objs.size() != n )
        return "Exactly " + std::to_string( n ) + " " + ObjectT::TypeName() + "(s) must be selected";
    for ( const auto& obj : objs )
    {
        auto tObj = dynamic_cast<const ObjectT*>( obj.get() );
        if ( !tObj )
            return std::string( "Selected object(s) must have type: " ) + ObjectT::TypeName();

        if constexpr ( modelCheck )
            if ( !tObj->hasModel() )
                return "Selected object(s) must have valid model";

        if constexpr ( visualRepresentationCheck )
            if ( !tObj->hasVisualRepresentation() )
                return "Selected object(s) must have valid visual representation";
    }
    return "";
}

// checks that given vector has at least N objects if type ObjectT
// returns error message if requirements are not satisfied
template<typename ObjectT, bool visualRepresentationCheck, bool modelCheck>
std::string sceneSelectedAtLeast( const std::vector<std::shared_ptr<const Object>>& objs, unsigned n )
{
    if ( objs.size() < n )
        return "At least " + std::to_string( n ) + " " + ObjectT::TypeName() + "(s) must be selected";
    unsigned i = 0;
    for ( const auto& obj : objs )
    {
        auto tObj = dynamic_cast<const ObjectT*>( obj.get() );
        if ( !tObj )
            continue;

        if constexpr ( modelCheck )
            if ( !tObj->hasModel() )
                continue;

        if constexpr ( visualRepresentationCheck )
            if ( !tObj->hasVisualRepresentation() )
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
        return sceneSelectedExactly<ObjectT, true, true>(objs, N);
    }
};

template<unsigned N, typename ObjectT>
class SceneStateExactCheck<N, ObjectT, NoVisualRepresentationCheck> : virtual public ISceneStateCheck
{
public:
    virtual ~SceneStateExactCheck() = default;
    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& objs ) const override
    {
        return sceneSelectedExactly<ObjectT, false, true>(objs, N);
    }
};

template<unsigned N, typename ObjectT>
class SceneStateExactCheck<N, ObjectT, NoModelCheck> : virtual public ISceneStateCheck
{
public:
    virtual ~SceneStateExactCheck() = default;
    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& objs ) const override
    {
        return sceneSelectedExactly<ObjectT, false, false>( objs, N );
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
        return sceneSelectedAtLeast<ObjectT, true, true>( objs, N );
    }
};

template<unsigned N, typename ObjectT>
class SceneStateAtLeastCheck<N, ObjectT, NoVisualRepresentationCheck> : virtual public ISceneStateCheck
{
public:
    virtual ~SceneStateAtLeastCheck() = default;
    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& objs ) const override
    {
        return sceneSelectedAtLeast<ObjectT, false, true>( objs, N );
    }
};

template<unsigned N, typename ObjectT>
class SceneStateAtLeastCheck<N, ObjectT, NoModelCheck> : virtual public ISceneStateCheck
{
public:
    virtual ~SceneStateAtLeastCheck() = default;
    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& objs ) const override
    {
        return sceneSelectedAtLeast<ObjectT, false, false>( objs, N );
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