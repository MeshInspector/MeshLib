#pragma once
#ifdef __EMSCRIPTEN__
#include "MRCommonPlugins/exports.h"
#include "MRViewer/MRStatePlugin.h"
#include "MRViewer/MRRibbonMenu.h"

namespace MR
{
class MRCOMMONPLUGINS_CLASS WasmUnavailablePlugin : public StatePlugin
{
public:
    WasmUnavailablePlugin( const std::string& name ) :
        StatePlugin( name ){}
    
    MRCOMMONPLUGINS_API virtual void drawDialog( float menuScaling, ImGuiContext* ) override;
    virtual bool blocking() const override { return false; }
private:
    virtual bool onEnable_() override { openPopup_ = true; return true; }
    bool openPopup_{true};
};


template<typename AvailabilityCheckClass>
class WasmUnavailablePluginWithSceneCheck : public WasmUnavailablePlugin, public AvailabilityCheckClass {
public:
    WasmUnavailablePluginWithSceneCheck( const std::string& name ) :
        WasmUnavailablePlugin( name ) {}
};

class WasmUnavailableObjectVoxels
{
public:
    constexpr static const char* TypeName() noexcept { return "ObjectVoxels"; }
};


}

#define MR_REGISTER_WASM_UNAVAILABLE_ITEM( pluginType, name )\
    static MR::RibbonMenuItemAdder<MR::WasmUnavailablePlugin> ribbonMenuItemAdder##pluginType##_(name);

#define MR_REGISTER_WASM_UNAVAILABLE_ITEM_CHECK( pluginType, availCheckType, name )\
    static MR::RibbonMenuItemAdder<MR::WasmUnavailablePluginWithSceneCheck<availCheckType>> ribbonMenuItemAdder##pluginType##_(name);

#endif
