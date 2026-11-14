#pragma once
#include "MRMesh/MRFeatureAlignment.h"
#include <MRMesh/MRObject.h>
#include <array>
#include <memory>

namespace MR::FeatureAlignmentTool
{
using FeatureAlignment::Axis;
using FeatureAlignment::AxisDirection;
using FeatureAlignment::CenterPair;
using FeatureAlignment::Constraints;
using FeatureAlignment::Hole;
using FeatureAlignment::HoleKind;
using FeatureAlignment::HoleMode;
using FeatureAlignment::HolePair;
using FeatureAlignment::HoleResidual;
using FeatureAlignment::NormalRelation;
using FeatureAlignment::Plane;
using FeatureAlignment::PrimaryMode;
using FeatureAlignment::Settings;
using FeatureAlignment::Solution;

using ObjectPtr = std::shared_ptr<Object>;
struct CatalogEntry
{
    ObjectPtr object;
    std::string path;
    bool part{};
};
using Catalog = std::vector<CatalogEntry>;
Catalog catalog( const ObjectPtr &root );
struct OrderedRoles
{
    ObjectPtr fixed, moving;
};
// The scene-tree preorder is authoritative: upper stays fixed, lower moves.
Expected<OrderedRoles> hierarchyRoles( const Catalog &entries, const ObjectPtr &first, const ObjectPtr &second );
Expected<OrderedRoles> selectedHierarchyRoles( const Catalog &entries );
bool below( const Object *object, const Object *ancestor );
bool isHole( const ObjectPtr &object );
bool isPlane( const ObjectPtr &object );
bool isDirection( const ObjectPtr &object );
bool isAxis( const ObjectPtr &object );
bool isCenter( const ObjectPtr &object );
struct Selection
{
    ObjectPtr moving, fixed, movingPrimary, fixedPrimary;
    std::array<ObjectPtr, 2> movingLocators, fixedLocators;
    ObjectPtr movingClock, fixedClock;
};
Expected<Constraints> readConstraints( const ObjectPtr &root, const Selection &selection, const Settings &settings );
Expected<ObjectPtr> oneSelectedPart( const Catalog &catalog );
struct NodeStamp
{
    ObjectPtr object;
    const Object *parent{};
    AffineXf3f local;
    const void *modelToken{};
    std::string name;
};
struct SceneStamp
{
    std::vector<NodeStamp> nodes;
    Expected<bool> matches( const ObjectPtr &root ) const;
};
Expected<SceneStamp> stamp( const ObjectPtr &root );
} // namespace MR::FeatureAlignmentTool
