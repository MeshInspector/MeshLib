#pragma once
#include "vector_types.h"

namespace MR
{
namespace Cuda
{

struct Box2
{
    float2 min;
    float2 max;
};

struct Node
{
    Box2 box;
    int l, r;
};


}
}
