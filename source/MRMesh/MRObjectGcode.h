#pragma once
#include "MRObjectLinesHolder.h"
#include "MRGcodeExecutor.h"
#include "MRColor.h"

namespace MR
{

using GcodeSource = std::vector<std::string>;

/// an object that stores a g-code
/// \ingroup DataModelGroup
class MRMESH_CLASS ObjectGcode : public ObjectLinesHolder
{
public:
    MRMESH_API ObjectGcode();
    ObjectGcode( ObjectGcode&& ) = default;
    ObjectGcode& operator=( ObjectGcode&& ) = default;
    virtual ~ObjectGcode() = default;

    constexpr static const char* TypeName() noexcept { return "ObjectGcode"; }
    virtual const char* typeName() const override { return TypeName(); }

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    MRMESH_API virtual void setGcodeSource( const std::shared_ptr<GcodeSource>& gcodeSource );

    virtual const std::shared_ptr<GcodeSource>& gcodeSource() const { return gcodeSource_; }

    MRMESH_API virtual void setDirtyFlags( uint32_t mask ) override;

    /// \note this ctor is public only for std::make_shared used inside clone()
    ObjectGcode( ProtectedStruct, const ObjectGcode& obj ) : ObjectGcode( obj ) {}

    MRMESH_API virtual std::vector<std::string> getInfoLines() const override;
    virtual std::string getClassName() const override { return "G-code"; }

    // set drawing feedrate as gradient of brightness
    MRMESH_API void setFeedrateGradient( bool feedrateGradient );
    bool getFeedrateGradient() const { return feedrateGradientEnabled_; }

    MRMESH_API void setColor( const Color& color, bool work = true );
    const Color& getColor( bool work = true ) const { return work ? workColor_ : idleColor_; }

    /// signal about gcode changing, triggered in setDirtyFlag
    using GcodeChangedSignal = boost::signals2::signal<void( uint32_t mask )>;
    GcodeChangedSignal gcodeChangedSignal;

protected:
    MRMESH_API ObjectGcode( const ObjectGcode& other );

    /// swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;
    /// swaps signals, used in `swap` function to return back signals after `swapBase_`
    /// pls call Parent::swapSignals_ first when overriding this function
    MRMESH_API virtual void swapSignals_( Object& other ) override;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    MRMESH_API void deserializeFields_( const Json::Value& root ) override;

private:
    std::shared_ptr<GcodeSource> gcodeSource_;
    std::vector<GcodeExecutor::MoveAction> actionList_;
    Color idleColor_ = Color(0.3f, 0.3f, 0.3f);
    Color workColor_ = Color( 0.f, 0.f, 1.f );
    float maxFeedrate_ = 0.f;
    bool feedrateGradientEnabled_ = true;

    void updateColors_();

};

}
