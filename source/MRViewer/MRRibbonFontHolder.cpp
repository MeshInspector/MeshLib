#include "MRRibbonFontHolder.h"

namespace MR
{

RibbonFontHolder::RibbonFontHolder( const RibbonFontManager::FontType& fontType )
{
    font_ = RibbonFontManager::getFontByTypeStatic( fontType );
    size_ = RibbonFontManager::getFontSizeByType( fontType );
    assert( size_ >= 1.f );
    pushFont();
}

RibbonFontHolder::RibbonFontHolder( const RibbonFontManager::FontType& fontType, float scale )
{
    font_ = RibbonFontManager::getFontByTypeStatic( fontType );
    size_ = RibbonFontManager::getFontSizeByType( fontType ) * scale;
    assert( size_ >= 1.f );
    pushFont();
}

RibbonFontHolder::RibbonFontHolder( const RibbonFontManager::FontType& fontType, float scale, bool pushOnCreate )
{
    font_ = RibbonFontManager::getFontByTypeStatic( fontType );
    size_ = RibbonFontManager::getFontSizeByType( fontType ) * scale;
    assert( size_ >= 1.f );
    if ( pushOnCreate )
        pushFont();
}

RibbonFontHolder::~RibbonFontHolder()
{
    assert( !pushed_ );
}

void RibbonFontHolder::pushFont()
{
    if ( !font_ )
        return;
    pushed_ = true;
    ImGui::PushFont( font_, size_ );
}

void RibbonFontHolder::popFont()
{
    if ( !font_ )
        return;
    pushed_ = false;
    ImGui::PopFont();
}

}
