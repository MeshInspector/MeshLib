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

        using MagnificationCallback = std::function<void ( float scale, bool finished )>;
        void onMagnification( MagnificationCallback cb );

        using RotationCallback = std::function<void ( float angle, bool finished )>;
        void onRotation( RotationCallback cb );

        using ScrollCallback = std::function<void ( float dx, float dy )>;
        void onMouseScroll( ScrollCallback cb );
        void onTouchScroll( ScrollCallback cb );

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
        MagnificationCallback magnificationCb_;
        RotationCallback rotationCb_;
        ScrollCallback mouseScrollCb_;
        ScrollCallback touchScrollCb_;
    };
}