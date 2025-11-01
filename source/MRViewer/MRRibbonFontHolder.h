#pragma once
#include "exports.h"
#include "MRRibbonFontManager.h" // TODO move it in MRRibbonFontManager.cpp

namespace MR
{

// class for convenient use of ImGui::PushFont / ImGui::PopFont with ribbon fonts
class MRVIEWER_CLASS RibbonFontHolder
{
public:
    // ctors
    // getting font and size by ribbon font type and push it in ImGui
    MRVIEWER_API RibbonFontHolder( const RibbonFontManager::FontType& fontType );
    MRVIEWER_API RibbonFontHolder( const RibbonFontManager::FontType& fontType, float scale );
    MRVIEWER_API RibbonFontHolder( const RibbonFontManager::FontType& fontType, float scale, bool pushOnCreate );
    MRVIEWER_API ~RibbonFontHolder();

    // check the font for existence and push it
    MRVIEWER_API void pushFont();

    // check the font is pushed and pop it
    MRVIEWER_API void popFont();

    // font is pushed
    bool isPushed() { return pushed_; }
private:
    bool pushed_ = false;
    ImFont* font_ = nullptr;
    float size_ = 13.f;
};
}
