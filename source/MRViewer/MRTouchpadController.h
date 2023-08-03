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

        enum class TouchState
        {
            Began,
            Moved,
            Ended,
            Canceled,
        };
        using TouchCallback = std::function<void ( size_t id, float x, float y, TouchState state )>;
        void onTouch( TouchCallback cb );

        class Impl
        {
        public:
            virtual ~Impl() = default;

            virtual void onMagnification( MagnificationCallback cb ) = 0;
            virtual void onRotation( RotationCallback cb ) = 0;
            virtual void onMouseScroll( ScrollCallback cb ) = 0;
            virtual void onTouchScroll( ScrollCallback cb ) = 0;
            virtual void onTouch( TouchCallback cb ) = 0;
        };

    private:
        std::unique_ptr<Impl> impl_;
    };
}