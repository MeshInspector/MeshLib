#pragma once

#include <MRMesh\MRViewportId.h>
#include "MRViewer/MRUIStyle.h"

#include <functional>

namespace MR
{
class CustomCheckBox
{
public:

    CustomCheckBox() = default;

    CustomCheckBox(
        std::function<void( std::shared_ptr<Object> object, ViewportId id, bool checked )> set,
        std::function<bool( std::shared_ptr<Object> object, ViewportId id )> get) :
        getter( get ),
        setter( set )
    {

    }

    std::function<bool( std::shared_ptr<Object> object, ViewportId id )> getter;
    std::function<void( std::shared_ptr<Object> object, ViewportId id, bool checked )> setter;

};

}
