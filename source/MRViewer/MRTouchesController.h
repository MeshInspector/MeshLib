#pragma once
#include "MRViewerFwd.h"
#include "MRViewerEventsListener.h"
#include "MRMesh/MRVector2.h"
#include "MRMouse.h"
#include <vector>
#include <optional>

namespace MR
{
// class to operate with touches
// only overrides signals
class MRVIEWER_CLASS TouchesController : public MultiListener<TouchStartListener,TouchMoveListener,TouchEndListener>
{
public:
    MR_ADD_CTOR_DELETE_MOVE( TouchesController );

    // bit meaning for mode mask
    enum ModeBit
    {
        Translate = 0b001,
        Rotate = 0b010,
        Zoom = 0b100,
        All = Zoom,
        Any = All
    };
    // mode mask can block some modes when two finger controll camera
    unsigned char getModeMask() const { return touchModeMask_; }
    void setModeMask( unsigned char mask ){ touchModeMask_ = mask; }
private:
    virtual bool onTouchStart_( int id, int x, int y ) override;
    virtual bool onTouchMove_( int id, int x, int y ) override;
    virtual bool onTouchEnd_( int id, int x, int y ) override;

    struct Info
    {
        int id{-1};
        Vector2f position;
    };

    class MultiInfo
    {
    public:
        bool update( Info info, bool remove = false );
        enum class Finger
        {
            First,
            Second
        };
        std::optional<Vector2f> getPosition( Finger fing ) const;
        std::optional<Vector2f> getPosition( int id ) const;
        std::optional<Finger> getFingerById( int id ) const;
        std::optional<int> getIdByFinger( Finger fing ) const;
        int getNumPressed() const;
    private:
        std::array<Info,2> info_;
    };

    MultiInfo multiInfo_;
    MultiInfo multiPrevInfo_;
    bool mouseMode_{ false };
    unsigned char touchModeMask_{ ModeBit::All };
};

}