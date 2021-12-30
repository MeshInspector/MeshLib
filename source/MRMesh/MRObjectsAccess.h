#pragma once
#include "MRObject.h"

namespace MR
{

enum class ObjectSelectivityType
{
    Selectable,
    Selected,
    Any
};

// if input object is of given type then returns another pointer on it
template<typename ObjectT = Object>
std::shared_ptr<ObjectT> asSelectivityType( std::shared_ptr<Object> obj, const ObjectSelectivityType& type );

// Traverses tree and collect objects of given type excluding root
// returns vector
template<typename ObjectT = Object>
std::vector<std::shared_ptr<ObjectT>> getAllObjectsInTree( Object* root, const ObjectSelectivityType& type = ObjectSelectivityType::Selectable );
template<typename ObjectT = Object>
inline std::vector<std::shared_ptr<ObjectT>> getAllObjectsInTree( Object& root, const ObjectSelectivityType& type = ObjectSelectivityType::Selectable )
    { return getAllObjectsInTree( &root, type ); }

// Returns all topmost visible objects of given type (if an object is returned, its children are not) excluding root
template<typename ObjectT = Object>
std::vector<std::shared_ptr<ObjectT>> getTopmostVisibleObjects( Object* root, const ObjectSelectivityType& type = ObjectSelectivityType::Selectable );
template<typename ObjectT = Object>
inline std::vector<std::shared_ptr<ObjectT>> getTopmostVisibleObjects( Object& root, const ObjectSelectivityType& type = ObjectSelectivityType::Selectable )
    { return getTopmostVisibleObjects( &root, type ); }

// return first object of given type in depth-first traverse order excluding root
template<typename ObjectT = Object>
std::shared_ptr<ObjectT> getDepthFirstObject( Object* root, const ObjectSelectivityType& type );
template<typename ObjectT = Object>
inline std::shared_ptr<ObjectT> getDepthFirstObject( Object& root, const ObjectSelectivityType& type )
    { return getDepthFirstObject( &root, type ); }

}

#include "MRObjectsAccess.hpp"