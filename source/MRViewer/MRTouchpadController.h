#pragma once

#include <functional>
#include <memory>

struct GLFWwindow;

namespace MR
{
    class TouchpadController
    {
    public:
        void initialize( GLFWwindow* window );

        using ZoomCallback = std::function<void ( float scale, bool finished )>;
        void onZoom( ZoomCallback cb );

        using RotateCallback = std::function<void ( float angle, bool finished )>;
        void onRotate( RotateCallback cb );

        using ScrollSwipeCallback = std::function<void ( float dx, float dy )>;
        void onMouseScroll( ScrollSwipeCallback cb );
        void onSwipe( ScrollSwipeCallback cb );

        class Impl
        {
        public:
            Impl( TouchpadController* controller, GLFWwindow* window );
            virtual ~Impl() = default;

            void mouseScroll( float dx, float dy );
            void rotate( float angle, bool finished );
            void swipe( float dx, float dy );
            void zoom( float scale, bool finished );

        private:
            TouchpadController* controller_;
        };

    private:
        std::unique_ptr<Impl> impl_;

        friend class Impl;
        ZoomCallback zoomCb_;
        RotateCallback rotateCb_;
        ScrollSwipeCallback mouseScrollCb_;
        ScrollSwipeCallback swipeCb_;
    };
}