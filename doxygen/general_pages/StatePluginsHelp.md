# State Plugins Overview {#StatePluginsHelp}

## Base Functions

**State plugins** is simple objects to work with MeshLib scene and/or objects. It is one of possible inheritance of \ref MR::RibbonMenuItem

It has base functions:

**Available function** - \ref MR::ISceneStateCheck interface function that block plugin in UI

> [!NOTE]
> This fuction can be inherited from some helper classes \ref MR::SceneStateExactCheck \ref MR::SceneStateAtLeastCheck \ref MR::SceneStateAtMostCheck \ref MR::SceneStateOrCheck \ref MR::SceneStateAndCheck

```cpp
    // return empty string if all requirements are satisfied, otherwise return first unsatisfied requirement
    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const { return ""; }
```
**On Enable function** - will be called if user enables plugin (only possible if `isAvailable` has returned true), if `onEnable_` function returns false plugin will not be enabled (useful for some expensive checks, not to do them in each frame `available` function)
```cpp
    MRVIEWER_API virtual bool onEnable_();
```
**On Disable function** - will be called if user disable plugin, it is necessary to clear all plugin data
```cpp
    MRVIEWER_API virtual bool onDisable_();
```
**Draw Dialog function** - this function is called in each frame, only if plugin is enabled (will not be called after `onDisable_` function). It is necessary to show plugin custom UI.
```cpp
    MRVIEWER_API virtual void drawDialog( ImGuiContext* ctx );
```

## How to make your first plugin
To create you plugin you need to inherit class from MR::StatePlugin or MR::StateListenerPlugin (in case you need to get some events)<br/>
Example:
```cpp
class MyOwnPlugin : public MR::StatePlugin
{
public:
	MyOwnPlugin():MR::StatePlugin("My Plugin Name"){}
	virtual std::string isAvailable(const std::vector<std::shared_ptr<const MR::Object>>& selectedObjects) const override
	{
		if ( selectedObjects.size() == 1 )
		    return "";
		return "Exactly one object should be selected";
	}
	virtual void drawDialog( ImGuiContext* ctx) override
	{
		auto windowSize = viewer->viewport().getViewportRect();
    	auto menuWidth = 280.0f * UI::scale();

    	ImGui::SetCurrentContext( ctx );
    	ImGui::SetNextWindowPos( ImVec2( windowSize.width - menuWidth, 0 ), ImGuiCond_FirstUseEver );
    	ImGui::SetNextWindowSize( ImVec2( menuWidth, 0 ), ImGuiCond_FirstUseEver );
    	ImGui::SetNextWindowSizeConstraints( ImVec2( menuWidth, -1.0f ), ImVec2( menuWidth, -1.0f ) );
    	ImGui::Begin( plugin_name.c_str(), &dialogIsOpen_, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize );

    	if (ImGui::Button("Some action"))
    		someAction_();

    	ImGui::End();
	}

private:
	std::shared_ptr<MR::Object> obj_;

	// some action of this plugin
	void someAction_();

	virtual bool onEnable_() override
	{
		obj_ = MR::getAllObjectsInTree<MR::Object>( &MR::SceneRoot::get(), MR::ObjectSelectivityType::Selected )[0];
		return true;
	}
	virtual bool onDisable_() override
	{
		obj_.reset();
		return true;
	}
}
```

Register your plugin with
```cpp
MR_REGISTER_RIBBON_ITEM( MyOwnPlugin )
```
To show this item on Ribbon menu, it should be present in MenuSchema.ui.json file (special page about these files are coming soon)
<br/>
\ref MR::StateListenerPlugin is more simple way to make plugin listen to some events:
```cpp
// This plugin will listen to mouse move and mouse down, also isAvailable function will be imlemented with SceneStateExactCheck
class MyListenerPlugin : public StateListenerPlugin<MouseMoveListener,MouseDownListener>, public SceneStateExactCheck<1, ObjectMesh>
```

Find more:
\ref MR::RibbonMenuItem
\ref MR::StateBasePlugin
\ref MR::StateListenerPlugin
