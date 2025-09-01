#pragma once

#include "MRObject.h"
#include "MRMeshTexture.h"
#include "MRVector.h"
#include "MRColor.h"
#include "MRIRenderObject.h"
#include "MRUniquePtr.h"
#include "MREnums.h"

namespace MR
{

/// \defgroup VisualObjectGroup Visual Object group
/// \ingroup DataModelGroup
/// \{

// Note! Must use `MRMESH_CLASS` on this enum and all enums that extend this one,
// otherwise you'll get silent wrong behavior on Mac.
enum class MRMESH_CLASS VisualizeMaskType
{
    Visibility,
    InvertedNormals,
    Name,
    ClippedByPlane,
    DepthTest,
    _count [[maybe_unused]],
};

// If a type derived from `VisualObject` wants to extend `VisualizeMaskType`, it must create a separate enum and specialize this to `true` for it.
// NOTE! All those enums can start from 0, don't worry about collisions.
template <typename T> struct IsVisualizeMaskEnum : std::false_type {};
template <> struct IsVisualizeMaskEnum<VisualizeMaskType> : std::true_type {};

// Wraps `IsVisualizeMaskEnum` and adds some sanity checks.
template <typename T>
concept AnyVisualizeMaskEnumType =
    IsVisualizeMaskEnum<T>::value &&
    std::is_same_v<std::underlying_type_t<T>, int> &&
    std::is_same_v<T, std::remove_cvref_t<T>>;

// Stores a `VisualizeMaskType` or any other enum that extends it (i.e. which specializes `IsVisualizeMaskEnum`).
// To extract the value, do this:
//     if ( auto value = x.tryGet<MyEnum>() )
//     {
//         switch ( *value )
//         {
//             case MyEnum::foo: ...
//             case MyEnum::bar: ...
//         }
//     }
//     else // forward to the parent class
class AnyVisualizeMaskEnum
{
    std::type_index type_;
    int value_ = 0;

public:
    template <AnyVisualizeMaskEnumType T>
    AnyVisualizeMaskEnum( T value ) : type_( typeid(T) ), value_( decltype(value_)( value ) ) {}

    template <AnyVisualizeMaskEnumType T>
    [[nodiscard]] std::optional<T> tryGet() const
    {
        if ( type_ == typeid(T) )
            return T( value_ );
        else
            return {};
    }
};

using AllVisualizeProperties = std::vector<ViewportMask>;

enum DirtyFlags
{
    DIRTY_NONE = 0x0000,
    DIRTY_POSITION = 0x0001,
    DIRTY_UV = 0x0002,
    DIRTY_VERTS_RENDER_NORMAL = 0x0004, //< gl normals
    DIRTY_FACES_RENDER_NORMAL = 0x0008, ///< gl normals
    DIRTY_CORNERS_RENDER_NORMAL = 0x0010, ///< gl normals
    DIRTY_RENDER_NORMALS = DIRTY_VERTS_RENDER_NORMAL | DIRTY_FACES_RENDER_NORMAL | DIRTY_CORNERS_RENDER_NORMAL,
    DIRTY_SELECTION = 0x0020,
    DIRTY_TEXTURE = 0x0040,
    DIRTY_PRIMITIVES = 0x0080,
    DIRTY_FACE = DIRTY_PRIMITIVES,
    DIRTY_VERTS_COLORMAP = 0x0100,
    DIRTY_PRIMITIVE_COLORMAP = 0x0200,
    DIRTY_FACES_COLORMAP = DIRTY_PRIMITIVE_COLORMAP,
    DIRTY_TEXTURE_PER_FACE = 0x0400,
    DIRTY_MESH = 0x07FF,
    DIRTY_BOUNDING_BOX = 0x0800,
    DIRTY_BORDER_LINES = 0x1000,
    DIRTY_EDGES_SELECTION = 0x2000,
    DIRTY_CACHES = DIRTY_BOUNDING_BOX,
    DIRTY_VOLUME = 0x4000,
    DIRTY_ALL = 0x7FFF
};

/// Marks dirty buffers that need to be uploaded to OpenGL.
/// Dirty flags must be moved together with renderObj_,
/// but not copied since renderObj_ is not copied as well
struct Dirty
{
    uint32_t f{DIRTY_ALL};
    operator uint32_t&() { return f; }
    operator uint32_t() const { return f; }

    Dirty() noexcept = default;
    Dirty( const Dirty& ) noexcept {}
    Dirty( Dirty&& ) noexcept = default;
    Dirty& operator =( const Dirty& ) noexcept { return *this; }
    Dirty& operator =( Dirty&& ) noexcept = default;
    Dirty& operator =( uint32_t b ) noexcept { f = b; return *this; }
};

/// Visual Object
class MRMESH_CLASS VisualObject : public Object
{
public:
    MRMESH_API VisualObject();

    VisualObject( VisualObject&& ) = default;
    VisualObject& operator = ( VisualObject&& ) = default;
    virtual ~VisualObject() = default;

    constexpr static const char* TypeName() noexcept { return "VisualObject"; }
    virtual const char* typeName() const override { return TypeName(); }

    constexpr static const char* ClassName() noexcept { return "Visual Object"; }
    virtual std::string className() const override { return ClassName(); }

    constexpr static const char* ClassNameInPlural() noexcept { return "Visual Objects"; }
    virtual std::string classNameInPlural() const override { return ClassNameInPlural(); }

    /// Returns true if this class supports the property `type`. Otherwise passing it to the functions below is illegal.
    [[nodiscard]] MRMESH_API virtual bool supportsVisualizeProperty( AnyVisualizeMaskEnum type ) const;
    /// set visual property in all viewports specified by the mask
    MRMESH_API void setVisualizeProperty( bool value, AnyVisualizeMaskEnum type, ViewportMask viewportMask );
    /// set visual property mask
    MRMESH_API virtual void setVisualizePropertyMask( AnyVisualizeMaskEnum type, ViewportMask viewportMask );
    /// returns true if the property is set at least in one viewport specified by the mask
    MRMESH_API bool getVisualizeProperty( AnyVisualizeMaskEnum type, ViewportMask viewportMask ) const;
    /// returns mask of viewports where given property is set
    MRMESH_API virtual const ViewportMask& getVisualizePropertyMask( AnyVisualizeMaskEnum type ) const;
    /// toggle visual property in all viewports specified by the mask
    MRMESH_API void toggleVisualizeProperty( AnyVisualizeMaskEnum type, ViewportMask viewportMask );

    /// get all visualize properties masks
    MRMESH_API virtual AllVisualizeProperties getAllVisualizeProperties() const;
    /// set all visualize properties masks
    void setAllVisualizeProperties( const AllVisualizeProperties& properties )
    {
        std::size_t counter = 0;
        setAllVisualizeProperties_( properties, counter );
    }

    /// returns all viewports where this object or any of its parents is clipped by plane
    [[nodiscard]] MRMESH_API ViewportMask globalClippedByPlaneMask() const;

    /// returns true if this object or any of its parents is clipped by plane in any of given viewports
    [[nodiscard]] bool globalClippedByPlane( ViewportMask viewportMask = ViewportMask::any() ) const { return !( globalClippedByPlaneMask() & viewportMask ).empty(); }

    /// if false deactivates clipped-by-plane for this object and all of its parents, otherwise sets clipped-by-plane for this this object only
    MRMESH_API void setGlobalClippedByPlane( bool on, ViewportMask viewportMask = ViewportMask::all() );

    /// shows/hides object name in all viewports
    void showName( bool on ) { return setVisualizeProperty( on, VisualizeMaskType::Name, ViewportMask::all() ); }
    /// returns whether object name is shown in any viewport
    bool showName() const { return getVisualizeProperty( VisualizeMaskType::Name, ViewportMask::any() ); }

    /// returns color of object when it is selected/not-selected (depending on argument) in given viewport
    MRMESH_API const Color& getFrontColor( bool selected = true, ViewportId viewportId = {} ) const;
    /// sets color of object when it is selected/not-selected (depending on argument) in given viewport
    MRMESH_API virtual void setFrontColor( const Color& color, bool selected, ViewportId viewportId = {} );

    /// returns color of object when it is selected/not-selected (depending on argument) in all viewports
    MRMESH_API virtual const ViewportProperty<Color>& getFrontColorsForAllViewports( bool selected = true ) const;
    /// sets color of object when it is selected/not-selected (depending on argument) in all viewports
    MRMESH_API virtual void setFrontColorsForAllViewports( ViewportProperty<Color> val, bool selected = true );

    /// returns backward color of object in all viewports
    MRMESH_API virtual const ViewportProperty<Color>& getBackColorsForAllViewports() const;
    /// sets backward color of object in all viewports
    MRMESH_API virtual void setBackColorsForAllViewports( ViewportProperty<Color> val );

    /// returns backward color of object in given viewport
    MRMESH_API const Color& getBackColor( ViewportId viewportId = {} ) const;
    /// sets backward color of object in given viewport
    MRMESH_API virtual void setBackColor( const Color& color, ViewportId viewportId = {} );

    /// returns global transparency alpha of object in given viewport
    MRMESH_API const uint8_t& getGlobalAlpha( ViewportId viewportId = {} ) const;
    /// sets global transparency alpha of object in given viewport
    MRMESH_API virtual void setGlobalAlpha( uint8_t alpha, ViewportId viewportId = {} );

    /// returns global transparency alpha of object in all viewports
    MRMESH_API virtual const ViewportProperty<uint8_t>& getGlobalAlphaForAllViewports() const;
    /// sets global transparency alpha of object in all viewports
    MRMESH_API virtual void setGlobalAlphaForAllViewports( ViewportProperty<uint8_t> val );

    /// sets some dirty flags for the object (to force its visual update)
    /// \param mask is a union of DirtyFlags flags
    /// \param invalidateCaches whether to automatically invalidate model caches (pass false here if you manually update the caches)
    MRMESH_API virtual void setDirtyFlags( uint32_t mask, bool invalidateCaches = true );
    /// returns current dirty flags for the object
    MRMESH_API uint32_t getDirtyFlags() const { return dirty_; }
    /// resets all dirty flags (except for cache flags that will be reset automatically on cache update)
    MRMESH_API void resetDirty() const;
    /// reset dirty flags without some specific bits (useful for lazy normals update)
    MRMESH_API virtual void resetDirtyExceptMask( uint32_t mask ) const;

    /// returns cached bounding box of this object in local coordinates
    MRMESH_API Box3f getBoundingBox() const;
    /// returns bounding box of this object in given viewport in world coordinates,
    /// to get world bounding box of the object with all child objects, please call Object::getWorldTreeBox method
    MRMESH_API virtual Box3f getWorldBox( ViewportId = {} ) const override;
    /// returns true if the object must be redrawn (due to dirty flags) in one of specified viewports
    virtual bool getRedrawFlag( ViewportMask viewportMask ) const override
    {
        return Object::getRedrawFlag( viewportMask ) ||
            ( isVisible( viewportMask ) &&
              ( dirty_ & ( ~( DIRTY_CACHES ) ) ) );
    }

    /// whether the object can be picked (by mouse) in any of given viewports
    bool isPickable( ViewportMask viewportMask = ViewportMask::any() ) const{return !(pickable_ & viewportMask).empty();}
    /// sets the object as can/cannot be picked (by mouse) in all of given viewports
    MRMESH_API virtual void setPickable( bool on, ViewportMask viewportMask = ViewportMask::all() );

    /// returns the current coloring mode of the object
    ColoringType getColoringType() const { return coloringType_; }

    /// sets coloring mode of the object with given argument
    MRMESH_API virtual void setColoringType( ColoringType coloringType );

    /// returns the current shininess visual value
    float getShininess() const { return shininess_; }
    /// sets shininess visual value of the object with given argument
    virtual void setShininess( float shininess ) { shininess_ = shininess; needRedraw_ = true; }

    /// returns intensity of reflections
    float getSpecularStrength() const { return specularStrength_; }
    /// sets intensity of reflections
    virtual void setSpecularStrength( float specularStrength ) { specularStrength_ = specularStrength; needRedraw_ = true; }

    /// returns intensity of non-directional light
    float getAmbientStrength() const { return ambientStrength_; }
    /// sets intensity of non-directional light
    virtual void setAmbientStrength( float ambientStrength ) { ambientStrength_ = ambientStrength; needRedraw_ = true; }

    /// clones this object only, without its children,
    /// making new object the owner of all copied resources
    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    /// clones this object only, without its children,
    /// making new object to share resources with this object
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    /// draws this object for visualization
    /// Returns true if something was drawn.
    MRMESH_API virtual bool render( const ModelRenderParams& ) const;
    /// draws this object for picking
    MRMESH_API virtual void renderForPicker( const ModelBaseRenderParams&, unsigned ) const;
    /// draws this object for 2d UI
    MRMESH_API virtual void renderUi( const UiRenderParams& params ) const;

    /// this ctor is public only for std::make_shared used inside clone()
    VisualObject( ProtectedStruct, const VisualObject& obj ) : VisualObject( obj ) {}

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API virtual size_t heapBytes() const override;
    /// return several info lines that can better describe the object in the UI
    MRMESH_API virtual std::vector<std::string> getInfoLines() const override;

    /// whether the scene-related properties should get their values from SceneColors and SceneSettings instances
    /// rather than from the input data on deserialization
    [[nodiscard]] MRMESH_API bool useDefaultScenePropertiesOnDeserialization() const { return useDefaultScenePropertiesOnDeserialization_; }
    /// set whether the scene-related properties should get their values from SceneColors and SceneSettings instances
    /// rather than from the input data on deserialization
    MRMESH_API void setUseDefaultScenePropertiesOnDeserialization( bool useDefaultScenePropertiesOnDeserialization )
    { useDefaultScenePropertiesOnDeserialization_ = useDefaultScenePropertiesOnDeserialization; }

    /// reset basic object colors to their default values from the current theme
    MRMESH_API virtual void resetFrontColor();
    /// reset all object colors to their default values from the current theme
    MRMESH_API virtual void resetColors();

protected:
    VisualObject( const VisualObject& obj ) = default;

    /// swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;

    /// each renderable child of VisualObject should implement this method
    /// and assign renderObj_ inside
    virtual void setupRenderObject_() const {}

    mutable UniquePtr<IRenderObject> renderObj_;

    /// Visualization options
    /// Each option is a binary mask specifying on which viewport each option is set.
    /// When using a single viewport, standard boolean can still be used for simplicity.
    ViewportMask clipByPlane_;
    ViewportMask showName_;
    ViewportMask pickable_ = ViewportMask::all(); ///< enable picking by gl
    ViewportMask invertNormals_; ///< invert mesh normals
    ViewportMask depthTest_ = ViewportMask::all();

    float shininess_{35.0f}; ///< specular exponent
    float specularStrength_{ 0.5f }; // reflection intensity
    float ambientStrength_{ 0.1f }; //non - directional light intensity

    /// Main coloring options
    ColoringType coloringType_{ColoringType::SolidColor};
    ViewportProperty<Color> selectedColor_;
    ViewportProperty<Color> unselectedColor_;
    ViewportProperty<Color> backFacesColor_;
    ViewportProperty<uint8_t> globalAlpha_{ 255 };

    bool useDefaultScenePropertiesOnDeserialization_{ false };

    MRMESH_API ViewportMask& getVisualizePropertyMask_( AnyVisualizeMaskEnum type );

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    MRMESH_API void deserializeFields_( const Json::Value& root ) override;

    virtual Box3f computeBoundingBox_() const { return Box3f(); }

    /// adds information about bounding box in res
    MRMESH_API void boundingBoxToInfoLines_( std::vector<std::string> & res ) const;

    MRMESH_API virtual void setAllVisualizeProperties_( const AllVisualizeProperties& properties, std::size_t& pos );

    // Derived classes should use this to implement `setAllVisualizeProperties()`.
    template <AnyVisualizeMaskEnumType T>
    void setAllVisualizePropertiesForEnum( const AllVisualizeProperties& properties, std::size_t& pos )
    {
        for ( int i = 0; i < int( T::_count ); i++ )
            setVisualizePropertyMask( T( i ), properties[pos++] );
    }
    // Derived classes should use this to implement `getAllVisualizeProperties()`.
    template <AnyVisualizeMaskEnumType T>
    void getAllVisualizePropertiesForEnum( AllVisualizeProperties& properties ) const
    {
        properties.reserve( properties.size() + std::size_t( T::_count ) );
        for ( int i = 0; i < int( ( T::_count ) ); i++ )
            properties.push_back( getVisualizePropertyMask( T( i ) ) );
    }

private:
    mutable Dirty dirty_; // private dirty, to force all using setDirtyFlags, instead of direct change

    mutable Box3f boundingBoxCache_;

    /// this is private function to set default colors of this type (Visual Object) in constructor only
    void setDefaultColors_();

    /// set default scene-related properties
    void setDefaultSceneProperties_();
};

/// \}

} // namespace MR
