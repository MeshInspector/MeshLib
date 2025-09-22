#pragma once
#include "MRObjectLinesHolder.h"
#include "MRGcodeProcessor.h"
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

    constexpr static const char* TypeName() noexcept { return "ObjectGcode"; }
    virtual const char* typeName() const override { return TypeName(); }

    constexpr static const char* ClassName() noexcept { return "G-code"; }
    virtual std::string className() const override { return ClassName(); }

    constexpr static const char* ClassNameInPlural() noexcept { return "G-codes"; }
    virtual std::string classNameInPlural() const override { return ClassNameInPlural(); }

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    MRMESH_API virtual void setCNCMachineSettings( const CNCMachineSettings& cncSettings );
    virtual const CNCMachineSettings& getCNCMachineSettings() const { return cncMachineSettings_; }

    MRMESH_API virtual void setGcodeSource( const std::shared_ptr<GcodeSource>& gcodeSource );
    virtual const std::shared_ptr<GcodeSource>& gcodeSource() const { return gcodeSource_; }
    // get action list (produced from g-code source)
    virtual const std::vector<GcodeProcessor::MoveAction>& actionList() const { return actionList_; }
    // get mapping of tool path polyline segment id to source line number of g-code source
    virtual const std::vector<int>& segmentToSourceLineMap() const { return segmentToSourceLineMap_; }

    MRMESH_API virtual void setDirtyFlags( uint32_t mask, bool invalidateCaches = true ) override;

    /// \note this ctor is public only for std::make_shared used inside clone()
    ObjectGcode( ProtectedStruct, const ObjectGcode& obj ) : ObjectGcode( obj ) {}

    MRMESH_API virtual std::vector<std::string> getInfoLines() const override;

    // set drawing feedrate as gradient of brightness
    MRMESH_API void switchFeedrateGradient( bool isFeedrateGradientEnabled );
    bool isFeedrateGradient() const { return feedrateGradientEnabled_; }

    MRMESH_API void setIdleColor( const Color& color );
    const Color& getIdleColor() const { return idleColor_; }

    /// signal about gcode changing, triggered in setDirtyFlag
    using GcodeChangedSignal = Signal<void( uint32_t mask )>;
    GcodeChangedSignal gcodeChangedSignal;

    MRMESH_API virtual bool select( bool isSelected ) override;
    MRMESH_API virtual void setFrontColor( const Color& color, bool selected, ViewportId viewportId = {} ) override;

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API virtual size_t heapBytes() const override;

protected:
    ObjectGcode( const ObjectGcode& other ) = default;

    /// swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;
    /// swaps signals, used in `swap` function to return back signals after `swapBase_`
    /// pls call Parent::swapSignals_ first when overriding this function
    MRMESH_API virtual void swapSignals_( Object& other ) override;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    MRMESH_API void deserializeFields_( const Json::Value& root ) override;

private:
    std::shared_ptr<GcodeSource> gcodeSource_;
    std::vector<GcodeProcessor::MoveAction> actionList_;
    CNCMachineSettings cncMachineSettings_;
    void updateHeapUsageCache_();
    size_t nonTrivialHeapUsageCache_{ 0 };

    std::vector<int> segmentToSourceLineMap_;

    Color idleColor_ = Color(0.3f, 0.3f, 0.3f);
    float maxFeedrate_ = 0.f;
    bool feedrateGradientEnabled_ = true;

    void updateColors_();
    void updateAll_();
};

}
