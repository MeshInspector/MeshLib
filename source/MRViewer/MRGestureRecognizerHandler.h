#pragma once

#include <functional>

struct GLFWwindow;

namespace MR
{
    class GestureRecognizerHandler
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
            virtual ~Impl() = default;

            virtual void onMagnification( MagnificationCallback cb ) = 0;
            virtual void onRotation( RotationCallback cb ) = 0;
            virtual void onMouseScroll( ScrollCallback cb ) = 0;
            virtual void onTouchScroll( ScrollCallback cb ) = 0;
        };

    private:
        std::unique_ptr<Impl> impl_;
    };
}