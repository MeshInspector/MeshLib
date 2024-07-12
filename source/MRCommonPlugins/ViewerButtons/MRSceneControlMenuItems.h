#pragma once

#include "MRViewer/MRRibbonMenuItem.h"
#include "MRViewer/MRHistoryStore.h"
#include "MRViewer/MRViewerEventsListener.h"

namespace MR
{

class ResetSceneMenuItem : public RibbonMenuItem, public MultiListener<PreDrawListener>
{
public:
   ResetSceneMenuItem();
   virtual bool action() override;
    virtual bool blocking()const override { return true; }
private:
    virtual void preDraw_() override;
    void resetScene_();
    bool openPopup_{ false };
    unsigned popupId_{ 0 };
};

class FitDataMenuItem : public RibbonMenuItem
{
public:
   FitDataMenuItem();
   virtual bool action() override;
   virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const override;
};

class FitSelectedObjectsMenuItem : public RibbonMenuItem
{
public:
   FitSelectedObjectsMenuItem();
   virtual bool action() override;
   virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const override;
};

class FitSelectedPrimitivesMenuItem : public RibbonMenuItem
{
public:
   FitSelectedPrimitivesMenuItem();
   virtual bool action() override;
   virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const override;
};

class SetViewPresetMenuItem : public RibbonMenuItem
{
public:
    enum class Type
    {
        Front,
        Top,
        // 2 is skipped
        Bottom = 3,
        Left,
        Back,
        Right,
        Isometric,
        Count
    };
   SetViewPresetMenuItem( Type type );
   virtual bool action() override;
private:
    Type type_;
};

class SetViewportConfigPresetMenuItem : public RibbonMenuItem
{
public:
    enum class Type
    {
        Single,
        Horizontal,
        Vertical,
        Quad,
        Hex,
        Count
    };
   SetViewportConfigPresetMenuItem( Type type );
   virtual void setCustomUpdateViewports( const std::function<void( const ViewportMask )>& callback ) { updateViewports_ = callback; }
   virtual bool action() override;
private:
    Type type_;
    std::function<void( const ViewportMask appendedViewports )> updateViewports_;
};

}