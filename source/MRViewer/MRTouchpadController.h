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
        void initialize( GLFWwindow* window );
        void connect();

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
}