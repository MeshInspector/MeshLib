#pragma once
#include "exports.h"
#include "MRRibbonFontManager.h" // TODO move it in MRRibbonFontManager.cpp

namespace MR
{

class MRVIEWER_CLASS RibbonFontHolder
{
public:
    MRVIEWER_API RibbonFontHolder( const RibbonFontManager::FontType& fontType );
    MRVIEWER_API RibbonFontHolder( const RibbonFontManager::FontType& fontType, float scale );
    MRVIEWER_API RibbonFontHolder( const RibbonFontManager::FontType& fontType, float scale, bool pushOnCreate );
    MRVIEWER_API ~RibbonFontHolder();

    MRVIEWER_API void pushFont();

    MRVIEWER_API void popFont();

    bool isPushed() { return pushed_; }
private:
    bool pushed_ = false;
    ImFont* font_ = nullptr;
    float size_ = 13.f;
};
}
