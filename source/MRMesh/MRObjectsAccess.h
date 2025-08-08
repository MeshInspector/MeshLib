#pragma once
#include "MRObject.h"
#include "MRPch/MRBindingMacros.h"

namespace MR
{

/// \ingroup DataModelGroup
/// \{

enum class MRMESH_CLASS ObjectSelectivityType
{
    Selectable,
    Selected,
    Any
};

/// if input object is of given type then returns another pointer on it
template<typename ObjectT = Object>
std::shared_ptr<ObjectT> asSelectivityType( std::shared_ptr<Object> obj, const ObjectSelectivityType& type );

/// Traverses tree and collect objects of given type excluding root
/// returns vector
template<typename ObjectT = Object>
std::vector<std::shared_ptr<ObjectT>> getAllObjectsInTree( Object* root, const ObjectSelectivityType& type = ObjectSelectivityType::Selectable );
template<typename ObjectT = Object>
MR_BIND_IGNORE inline std::vector<std::shared_ptr<ObjectT>> getAllObjectsInTree( Object& root, const ObjectSelectivityType& type = ObjectSelectivityType::Selectable )
    { return getAllObjectsInTree<ObjectT>( &root, type ); }

/// Returns all topmost visible objects of given type (if an object is returned, its children are not) excluding root
template<typename ObjectT = Object>
std::vector<std::shared_ptr<ObjectT>> getTopmostVisibleObjects( Object* root, const ObjectSelectivityType& type = ObjectSelectivityType::Selectable )
    { return getTopmostObjects<ObjectT>( root, type, true ); }
template<typename ObjectT = Object>
MR_BIND_IGNORE inline std::vector<std::shared_ptr<ObjectT>> getTopmostVisibleObjects( Object& root, const ObjectSelectivityType& type = ObjectSelectivityType::Selectable )
    { return getTopmostObjects<ObjectT>( &root, type, true ); }

/// Returns all topmost objects of given type (if an object is returned, its children are not) excluding root
template<typename ObjectT = Object>
std::vector<std::shared_ptr<ObjectT>> getTopmostObjects( Object* root, const ObjectSelectivityType& type = ObjectSelectivityType::Selectable, bool visibilityCheck = false );
template<typename ObjectT = Object>
MR_BIND_IGNORE inline std::vector<std::shared_ptr<ObjectT>> getTopmostObjects( Object& root, const ObjectSelectivityType& type = ObjectSelectivityType::Selectable, bool visibilityCheck = false )
    { return getTopmostObjects<ObjectT>( &root, type, visibilityCheck ); }

/// return first object of given type in depth-first traverse order excluding root
template<typename ObjectT = Object>
std::shared_ptr<ObjectT> getDepthFirstObject( Object* root, const ObjectSelectivityType& type );
template<typename ObjectT = Object>
MR_BIND_IGNORE inline std::shared_ptr<ObjectT> getDepthFirstObject( Object& root, const ObjectSelectivityType& type )
    { return getDepthFirstObject<ObjectT>( &root, type ); }

/// \}

inline bool objectHasSelectableChildren( const MR::Object& object )
{
    for ( const auto& child : object.children() )
    {
        if ( !child->isAncillary() || objectHasSelectableChildren( *child ) )
            return true;
    }
    return false;
}

}

#include "MRObjectsAccess.hpp"
