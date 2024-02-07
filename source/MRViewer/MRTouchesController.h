#pragma once
#include "MRMesh/MRFlagOperators.h"
#include "MRViewerFwd.h"
#include "MRViewerEventsListener.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMouse.h"
#include <vector>
#include <optional>
#include <functional>

namespace MR
{
// class to operate with touches
// only overrides signals
class MRVIEWER_CLASS TouchesController : public MultiListener<TouchStartListener,TouchMoveListener,TouchEndListener>
{
public:
    MR_ADD_CTOR_DELETE_MOVE( TouchesController );

    // set callback to modify view transform before it is applied to viewport
    void setTrasformModifierCb( std::function<void( AffineXf3f& )> cb ) { transformModifierCb_ = cb; }

    // bit meaning for mode mask
    enum class ModeBit : unsigned char
    {
        Translate = 0b001,
        Rotate = 0b010,
        Zoom = 0b100,
        All = Translate | Rotate | Zoom,
        Any = All
    };
    MR_MAKE_FLAG_OPERATORS_IN_CLASS( ModeBit )

    // mode mask can block some modes when two finger controll camera
    ModeBit getModeMask() const { return touchModeMask_; }
    void setModeMask( ModeBit mask ){ touchModeMask_ = mask; }
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
    ModeBit touchModeMask_{ ModeBit::All };

    std::function<void( AffineXf3f& )> transformModifierCb_;
};

}
