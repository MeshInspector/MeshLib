#pragma once

#include "MRMesh/MRMeshFwd.h"
#include "MRViewer/exports.h"

namespace MR
{

// This is used to temporarily change object colors (and other visual properties) in the scene.
// NOTE: Currently at most one highlighter can operate at a time.
// The workflow is as follows:
// 1. Create `ObjectHighlighter` as a class member in your plugin.
// 2. Using the constructor parameters, you can customize which fields we roll back. The default behavior should usually be good enough.
// 3. Call `updateColors()` every frame in `preDraw_()` to set the desired colors.
// 4. Call `restoreOriginalState()` in `onDisable_()`.
class ObjectHighlighter
{
public:
    // This is used to roll the object back to its original state.
    using RollbackFunc = std::function<void( Object& object )>;
    // Given an object, this saves its original parameters into a functor for a later rollback.
    // Can return null if nothing to save.
    using RuleFunc = std::function<RollbackFunc( const Object& object )>;

    // Predefined `RuleFunc`s for various object types: [
    MRVIEWER_API static RollbackFunc ruleObjectVisibility( const Object& object );
    MRVIEWER_API static RollbackFunc ruleVisualObjectColors( const Object& object );
    MRVIEWER_API static RollbackFunc ruleFeatures( const Object& object );
    // ]

    MRVIEWER_API ObjectHighlighter( std::vector<RuleFunc> rules = { ruleObjectVisibility, ruleVisualObjectColors, ruleFeatures } );

    using ModifyFunc = std::function<Object&( std::string name )>;

    // Call this once per frame, probably in `preDraw_()`.
    // If you want to modify `object`, call `modify("...")` and act on the returned reference. Otherwise do nothing.
    // The string passed to `modify` can be anything, but when it changes, modify resets the object to its original state before giving it back to you.
    // You can call `modify()` more than once, it doesn't matter.
    // NOTE! You probably shouldn't read the same parameter before modifying it, to avoid the effect from accumulating over several frames.
    MRVIEWER_API void highlight( std::function<void( const std::shared_ptr<const Object>& object, ModifyFunc modify )> highlightObject );

    // Call this when done, probably in `onDisable_()`, to restore all objects to their original states.
    MRVIEWER_API void restoreOriginalState();

private:
    // Those are used to figure out what information to store
    std::vector<RuleFunc> rules_;

    struct Entry
    {
        RollbackFunc func;
        std::string name;
    };

    // Those are used to restore the objects to their original states.
    std::unordered_map<std::shared_ptr<Object>, Entry> rollbackEntries_;
};

}
