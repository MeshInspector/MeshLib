#pragma once
#include "MRMesh/MRFeatureAlignment.h"
#include <MRMesh/MRObject.h>
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace MR::FeatureAlignmentTool
{
using FeatureAlignment::AxisDirection;
using FeatureAlignment::Constraints;
using FeatureAlignment::HoleMode;
using FeatureAlignment::NormalRelation;
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
// Uses the supplied selection, without cataloging unrelated scene branches.
Expected<OrderedRoles> selectedHierarchyRoles( const ObjectPtr &root,
    const std::vector<std::shared_ptr<const Object>> &selected );
bool haveCompatiblePrimaryFeatures( const OrderedRoles &roles );
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
    bool explicitRoles{};
};
Expected<Constraints> readConstraints( const ObjectPtr &root, const Selection &selection, const Settings &settings );
// Internal UI reuse only; mutating controller operations use the fresh-root overload.
Expected<Constraints> readConstraints( const ObjectPtr &root, const Catalog &entries,
    const Selection &selection, const Settings &settings );
} // namespace MR::FeatureAlignmentTool
