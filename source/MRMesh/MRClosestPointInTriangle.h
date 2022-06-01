// ======================================================================== //
// Copyright 2009-2020 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //
#pragma once

#include "MRVector3.h"
#include "MRTriPoint.h"

namespace MR
{

static std::pair<Vector3f, TriPointf> closestPointInTriangle( const Vector3f& p, const Vector3f& a, const Vector3f& b, const Vector3f& c )
{
    const Vector3f ab = b - a;
    const Vector3f ac = c - a;
    const Vector3f ap = p - a;

    const float d1 = dot( ab, ap );
    const float d2 = dot( ac, ap );
    if ( d1 <= 0.f && d2 <= 0.f ) 
        return { a, { 0, 0 } };

    const Vector3f bp = p - b;
    const float d3 = dot( ab, bp );
    const float d4 = dot( ac, bp );
    if ( d3 >= 0.f && d4 <= d3 ) 
        return { b, { 1, 0 } };

    const Vector3f cp = p - c;
    const float d5 = dot( ab, cp );
    const float d6 = dot( ac, cp );
    if ( d6 >= 0.f && d5 <= d6 ) 
        return { c, { 0, 1 } };

    const float vc = d1 * d4 - d3 * d2;
    if ( vc <= 0.f && d1 >= 0.f && d3 <= 0.f )
    {
        const float v = d1 / ( d1 - d3 );
        return { a + v * ab, { v, 0 } };
    }

    const float vb = d5 * d2 - d1 * d6;
    if ( vb <= 0.f && d2 >= 0.f && d6 <= 0.f )
    {
        const float v = d2 / ( d2 - d6 );
        return { a + v * ac, { 0, v } };
    }

    const float va = d3 * d6 - d5 * d4;
    if ( va <= 0.f && ( d4 - d3 ) >= 0.f && ( d5 - d6 ) >= 0.f )
    {
        const float v = ( d4 - d3 ) / ( ( d4 - d3 ) + ( d5 - d6 ) );
        return { b + v * ( c - b ), { 1 - v, v } };
    }

    const float denom = 1.f / ( va + vb + vc );
    const float v = vb * denom;
    const float w = vc * denom;
    return { a + v * ab + w * ac, { v, w } };
}

}
