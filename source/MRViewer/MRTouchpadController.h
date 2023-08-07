#pragma once

#include "MRViewport.h"

#include <functional>
#include <memory>

struct GLFWwindow;

namespace MR
{

class TouchpadController
{
public:
    MR_ADD_CTOR_DELETE_MOVE( TouchpadController );
    MRVIEWER_API void initialize( GLFWwindow* window );
    MRVIEWER_API void connect();

    struct Parameters
    {
        /// most touchpads implement kinetic (or inertial) scrolling, this option disables handling of these events
        bool ignoreKineticMoves = true;
        /// scale coefficient for swipe movements
        float swipeScale = 10.f;
    };
    MRVIEWER_API const Parameters& getParameters() const;
    MRVIEWER_API void setParameters( const Parameters& parameters );

    class Impl
    {
    public:
        virtual ~Impl() = default;

        enum class GestureState
        {
            Begin,
            Change,
            End,
            Cancel,
        };

        void mouseScroll( float dx, float dy, bool kinetic );
        void rotate( float angle, GestureState state );
        void swipe( float dx, float dy, bool kinetic );
        void zoom( float scale, GestureState state );
    };

private:
    std::unique_ptr<Impl> impl_;
    Parameters parameters_;

    Viewport::Parameters initRotateParams_;
    bool rotateStart_( float angle );
    bool rotateChange_( float angle );
    bool rotateCancel_();
    bool rotateEnd_();

    bool swipe_( float deltaX, float deltaY, bool kinetic );

    Viewport::Parameters initZoomParams_;
    bool zoomStart_( float scale );
    bool zoomChange_( float scale );
    bool zoomCancel_();
    bool zoomEnd_();
};

} // namespace MR