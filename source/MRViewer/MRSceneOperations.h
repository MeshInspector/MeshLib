#pragma once

#include "MRViewerFwd.h"

namespace MR
{

/// flat representation of an object tree
/// useful for caching
struct FlatTree
{
    std::shared_ptr<Object> root;
    std::vector<std::shared_ptr<Object>> subobjects;
};

/// flat representation of an object tree with objects being grouped by their types
/// to keep the class reasonably simple and useful, only mesh, polyline, and point cloud objects are grouped
struct TypedFlatTree
{
    std::shared_ptr<Object> root;
    std::vector<std::shared_ptr<ObjectMesh>> objsMesh;
    std::vector<std::shared_ptr<ObjectLines>> objsLines;
    std::vector<std::shared_ptr<ObjectPoints>> objsPoints;

    MRVIEWER_API static TypedFlatTree fromFlatTree( const FlatTree& tree );
};

/// get list of subtrees satisfying any of the following rules:
///  - all the subtree elements are present in the given object list
///  - only the subtree's root element is present in the given object list
/// TODO: optional predicate to ignore insignificant objects (non-visual objects, ancillary objects, etc.)
MRVIEWER_API std::vector<FlatTree> getFlatSubtrees( const std::vector<std::shared_ptr<Object>>& objs );

/// merge objects of same type in the object tree
MRVIEWER_API void mergeSubtree( TypedFlatTree subtree );
MRVIEWER_API void mergeSubtree( std::shared_ptr<Object> rootObj );

} // namespace MR
