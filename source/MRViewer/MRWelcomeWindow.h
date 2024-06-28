#pragma once
#include "exports.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector2.h"

namespace MR
{

class RibbonMenu;
class RibbonMenuItem;

// class to show Welcome window
class WelcomeWindow
{
public:
    // main draw function
    MRVIEWER_API void draw();

    // initialize internal variable
    MRVIEWER_API void init( std::function<void( const std::shared_ptr<RibbonMenuItem>&, bool )> itemPressedFn );

    MRVIEWER_API void setShowOnStartup( bool show, bool applyNow = false );
    bool getShowOnStartup() { return showOnStartup_; }
private:
    void drawDragDropArea_();
    void drawQuickstart_();
    void drawCreateSimpleObject_();
    void drawCheckbox_();

    // helping methods
    bool beginSubwindow_( const char* name, Vector2f pos, Vector2f size );
    
    bool showOnStartup_ = true;
    bool visible_ = true;

    std::shared_ptr<RibbonMenu> menu_;
    float scaling_ = 1.f;
    std::function<void( const std::shared_ptr<RibbonMenuItem>&, bool )> itemPressedFn_ = {};
    Vector2f sceneCenter_;
};

}
