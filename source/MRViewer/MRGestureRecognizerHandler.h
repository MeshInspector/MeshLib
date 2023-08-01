#pragma once

#include <functional>

struct GLFWwindow;

namespace MR
{
    class GestureRecognizerHandler
    {
    public:
        void initialize( GLFWwindow* window );

        using RotationCallback = std::function<void ( float angle )>;
        void onRotation( RotationCallback cb );

        class Impl
        {
        public:
            virtual ~Impl() = default;

            virtual void onRotation( RotationCallback cb ) = 0;
        };

    private:
        std::unique_ptr<Impl> impl_;
    };
}