#pragma once

#include "MRISceneStateCheck.h"
#include "MRMesh/MRObject.h"

namespace MR
{

// special struct for disabling visual representation check
struct NoVisualRepresentationCheck {};

// special struct for disabling model check
struct NoModelCheck {};

template<typename ObjectT>
std::string getNObjectsLine( unsigned n )
{
    std::string typeName = ObjectT::TypeName();
    if ( typeName.starts_with( "Object" ) && !typeName.ends_with( "Object" ) )
        typeName = typeName.substr( 6 );
    if ( typeName == "Points" )
        typeName = "Point Cloud";
    else if ( typeName == "Lines" )
        typeName = "Polyline";
    else if ( typeName == "Voxels" )
        typeName = "Volume";

    if ( n != 1 )
    {
        if ( typeName.ends_with( "s" ) || typeName.ends_with( "sh" ) )
            typeName += "es";
        else
            typeName += "s";
    }

    switch ( n )
    {
    case 1:
        return "one " + typeName;
    case 2:
        return "two " + typeName;
    case 3:
        return "three " + typeName;
    case 4:
        return "four " + typeName;
    default:
        return std::to_string( n ) + " " + typeName;
    }    
}

// check that given vector has exactly N objects if type ObjectT
// returns error message if requirements are not satisfied
template<typename ObjectT, bool visualRepresentationCheck, bool modelCheck>
std::string sceneSelectedExactly( const std::vector<std::shared_ptr<const Object>>& objs, unsigned n )
{
    if ( objs.size() != n )
        return "Select exactly " + getNObjectsLine<ObjectT>( n );
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
        return "Select at least " + getNObjectsLine<ObjectT>( n );
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
        ( "Select at least " + getNObjectsLine<ObjectT>( n ) + " with valid model" );
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
class SceneStateOrCheck : public Checks...
{
public:
    virtual ~SceneStateOrCheck() = default;
    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>&objs ) const override
    {
        std::vector<std::string> checkRes;
        checkRes.reserve( sizeof...( Checks ) );
        ( checkRes.push_back( Checks::isAvailable( objs ) ), ... );
        std::string combinedRes;
        for ( int i = 0; i  < checkRes.size(); ++i )
        {
            if ( checkRes[i].empty() )
                return "";

            if ( i != 0 )
                checkRes[i].front() = ( char )tolower( checkRes[i].front() );

            combinedRes += checkRes[i];
            if ( i + 1 < checkRes.size() )
                combinedRes += " or ";
        }
        return combinedRes;
    }
};

// checks that all of argument checks are true
template<typename ...Checks>
class SceneStateAndCheck : public Checks...
{
public:
    virtual ~SceneStateAndCheck() = default;
    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& objs ) const override
    {
        std::vector<std::string> checkRes;
        checkRes.reserve( sizeof...( Checks ) );
        ( checkRes.push_back( Checks::isAvailable( objs ) ), ... );
        std::string combinedRes;
        for ( int i = 0; i < checkRes.size(); ++i )
        {
            if ( checkRes[i].empty() )
                continue;

            if ( !combinedRes.empty() )
            {
                combinedRes += " and ";
                checkRes[i].front() = ( char )tolower( checkRes[i].front() );
            }
            combinedRes += checkRes[i];
        }
        return combinedRes;
    }
};


}