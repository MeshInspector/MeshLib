#pragma once

#include "MRMesh/MRFeatureObject.h"
#include "MRMesh/MRFeatures.h"
#include "MRViewer/MRObjectHighlighter.h"
#include "MRViewer/MRViewerEventsListener.h"
#include "MRViewer/MRViewport.h"
#include "MRViewer/exports.h"

#include <span>

namespace MR
{

// A helper class to pick into subfeatures.
// This is the minimal GUI-less version. Use `SubfeaturePickerWidget` to also get GUI and object highlighting.
// Creates temporary feature objects for the subfeatures and lets you pick into them.
// The workflow is:
// 1. Construct this as a member variable in your plugin.
// 2. Call `updateTemporaryObjects()` in `preDraw_()`.
// 3. Call `removeTemporaryObjects()` in `onDisable_()`.
// 4. Call `pickObject()` when you need it. If you do it from `mouseMove_()`, consult the comments on `pickObject()`.
//    Alternatively, you can use `temporarilyAdjustSubfeaturesForPicking()` to run your own customized picker.
//    You can do anything with the objects you receive from the picker. Subfeatures are invisible by default,
//      but you can make the picked one visible manually.
//    You can analyze the picked subfeatures using `findSubfeature()`.
class SubfeaturePickerWidgetLow
{
public:
    using ObjectPred = std::function<bool( const FeatureObject& feature, bool isAncillary )>;

    // If set, this is called for every new subfeature object we create.
    std::function<void( FeatureObject& )> initializeSubfeature;

    // Ignore featues that have `Subfeatures` visual property disabled.
    bool respectSubfeatureVisualProperty = true;
    // Whether ancillary features should be processed.
    bool allowAncillaryObjects = false;

    MRVIEWER_API static float defaultInfiniteExtent();
    // How large we should make "infinite" features.
    std::function<float()> getInfiniteExtent = defaultInfiniteExtent;

    // Temporarily override feature sizes to those values for picking:
    float pickerPointSize = 16;
    float pickerLineWidth = 10;

    struct SubfeatureData
    {
        std::string name;
        Features::Primitives::Variant subfeature;
        std::shared_ptr<FeatureObject> object;
    };

    struct ObjectData
    {
        Features::Primitives::Variant feature;
        std::vector<SubfeatureData> subfeatures;
    };

    using ObjectMap = std::map<std::shared_ptr<Object>, std::shared_ptr<ObjectData>>;
    // Returns all tracked feature objects.
    [[nodiscard]] MRVIEWER_API const ObjectMap& getObjects() const;

    // Update the temporary objects representing subfeatures.
    MRVIEWER_API void updateTemporaryObjects( ObjectPred objectPredicate = nullptr );

    // Remove all temporary objects. The next `updateTemporaryObjects()` will add them back.
    // Do this before disabling your plugin.
    MRVIEWER_API void removeTemporaryObjects();

    // Runs the picker for all visible pickable objects, including subfeatures.
    // Uses `temporarilyAdjustSubfeaturesForPicking()` under the hood.
    // NOTE! If you're going to call this from `mouseMove`, don't forget `if ( ImGui::GetIO().WantCaptureMouse ) /*pick nothing*/;` in `drawDialog_()`.
    [[nodiscard]] MRVIEWER_API ObjAndPick pickObject( std::function<bool( const Object& )> pred = nullptr );

    // Temporarily unhides the subfeature objects that are normally hidden, and also temporarily sets their radius and line width to improve picking.
    // Then calls `func()` and reverts everything back..
    // You normally don't need this funcion, prefer `pickObject()` if possible.
    MRVIEWER_API void temporarilyAdjustSubfeaturesForPicking( std::function<void()> func );

    struct FindResult
    {
        // Both bools can only be null at the same time.
        const ObjectMap::value_type* feature = nullptr;
        const SubfeatureData* subfeature = nullptr;

        [[nodiscard]] explicit operator bool() const { return feature && subfeature; }
    };

    // If `subfeature` is one of our subfeatures, returns the information about it, including the parent feature. Otherwise returns null.
    [[nodiscard]] MRVIEWER_API FindResult findSubfeature( const Object& subfeature );


private:
    // Using `std::shared_ptr` for the key here to be able to attach signal lifetimes to this.
    ObjectMap objects_;
};

// A helper class to pick into subfeatures.
// Highlights hovered objects, and/or draws GUI to further descend into subfeatures.
// Workflow:
// 1. Add this as a data member to your plugin. Also add `ObjectHighlighter`.
// 2. Call `enable()` and `disable()` in `onEnable_()` and `onDisable_()` respectively (or somewhere else, up to you).
// 3. Call `drawGui()` on `drawDialog_()`. You must do this even if you don't want any custom GUI (set `enableGui = false` to disable GUI).
//    If it returns true, we switched to a different subfeature, recalculate the result. (This changes the last element in the `targets` vector.)
// 4. In `preDraw_()`, pass `highlight` to `ObjectHighlighter::highlight()`. (Optional, don't do this to disable highlighting for hovered objects.)
// 5. In `mouseDown_()`, call `addTargetFromHover()`. On success this appends to the `targets` vector and returns true, then recalculate the result.
// The default behavior to add more objects to the `targets` when you call `addTargetFromHover()`, but you can freely manipulate the `targets` vector,
//   e.g. to keep only the last N objects.
// We highlight every object in this vector, and display the subfeature GUI for the last one.
class MRVIEWER_CLASS SubfeaturePickerWidget : MultiListener<PreDrawListener, MouseMoveListener>
{
public:
    SubfeaturePickerWidgetLow underlyingPicker;

    // Redundant enable/disable calls have no effect.
    MRVIEWER_API void enable();
    MRVIEWER_API void disable();
    [[nodiscard]] bool isEnabled() const { return isEnabled_; }

    // If this is false, then `drawGui()` doesn't draw any GUI (but still does some internal stuff).
    bool enableGui = true;

    // Call this from your `drawDialog_()`.
    // Returns true if the target object was changed.
    // Calls `preDrawGui` once before drawing any GUI, or not at all if we don't need to draw anything.
    MRVIEWER_API bool drawGui( std::function<void()> preDrawGui = {} );

    // This is for `ObjectHighlighter`. Returns true if this object was modified.
    MRVIEWER_API bool highlight( const std::shared_ptr<const Object>& object, ObjectHighlighter::ModifyFunc modify );

    // Returns the object hovered in the last `mouseMove_()`.
    [[nodiscard]] const ObjAndPick& getHoveredObject() const { return mouseMovePick_; }

    struct TargetStackEntry
    {
        std::string name;

        // In work coodinates, as always.
        Features::Primitives::Variant feature;

        // If this is a feature, points to the feature object.
        // If this is a subfeature clicked in the scene, points to the temporary subfeature object.
        // If this is a subfeature selected from GUI (that doesn't have a corresponding scene object), this is null.
        // If this is a clicked point/edge on a mesh, this is null.
        std::shared_ptr<Object> object;
    };
    struct Target
    {
        // The last element here is our target.
        // Should never be empty.
        std::vector<TargetStackEntry> stack;

        // The parent of the selected feature (when a subfeature is selected, this is the parent of its parent feature).
        // Or if a point/edge were selected, this is the object owning them.
        std::shared_ptr<Object> parentObject;
    };
    std::vector<Target> targets;

    // Appends to `targets`, adding a single feature.
    // Returns true on success (shouldn't normally fail).
    MRVIEWER_API bool addTargetFromFeature( const std::shared_ptr<FeatureObject>& feature );
    // Appends to `targets`, adding a single point on a mesh.
    MRVIEWER_API void addTargetFromPoint( const std::shared_ptr<Object>& object, Vector3f localPoint );

    enum class NonFeatureMode
    {
        // Use the clicked point, don't snap to vertices.
        point,
        // Clicked edge.
        // edge,
    };
    // How to handle clicking non-features.
    NonFeatureMode nonFeatureMode = NonFeatureMode::point;

    // If false, `addTargetFromHover()` will reject duplicate targets.
    // Note that it's still possible to select effectively duplicate targets using the subfeatures menu.
    bool allowDuplicateTargets = false;

    // Allow clicking features directly. (Default true unless Alt is held.)
    std::function<bool()> allowFeatures;
    [[nodiscard]] bool shouldAllowFeatures() const { return allowFeatures ? allowFeatures() : !disableFeatureHoverModifierIsHeld_; }

    // Allow selecting points/edges directly on objects. (Default true.)
    std::function<bool()> allowNonFeatures;
    [[nodiscard]] bool shouldAllowNonFeatures() const { return allowNonFeatures ? allowNonFeatures() : true; }

    // Overwrites `targetStack` based on the current hover (as set by the latest `mouseMove_()`).
    // Returns true on success.
    MRVIEWER_API bool addTargetFromHover();

    // Draws the subfeature selector gui for an arbitrary stack. `drawGui()` calls this for `targetStack`.
    // Returns true if the stack is changed. Will never make it empty.
    // Calls `preDrawGui` once before drawing any GUI, or not at all if we don't need to draw anything.
    MRVIEWER_API static bool drawSubfeatureGuiFor( std::vector<TargetStackEntry>& stack, std::function<void()> preDrawGui = {} );

private:
    MRVIEWER_API void preDraw_() override;
    MRVIEWER_API bool onMouseMove_( int x, int y ) override;

    bool isEnabled_ = false;

    ObjectHighlighter objectHighlighter_;

    ObjAndPick mouseMovePick_;

    struct TemporaryObject
    {
        std::shared_ptr<FeatureObject> object;
        std::optional<Features::Primitives::Variant> feature;
    };
    std::vector<TemporaryObject> temporaryObjects_;

    bool disableFeatureHoverModifierIsHeld_ = false;
};

}
