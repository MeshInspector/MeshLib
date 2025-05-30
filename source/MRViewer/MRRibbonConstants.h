#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRColor.h"

namespace
{
inline const std::string cPalettePresetKey = "palettePreset";
}

namespace MR
{
constexpr float cGradientButtonFramePadding = 7.5f;

constexpr float cTabYOffset = 4.0f;
constexpr float cTabsInterval = 8.0f;
constexpr float cTabFrameRounding = 5.0f;
constexpr float cTabMinimumWidth = 68.0f;
constexpr float cTabHeight = 28.0f;
constexpr float cTabLabelMinPadding = 12.0f;
constexpr float cTopPanelScrollStep = 50.0f;
constexpr float cTopPanelScrollBtnSize = 20.0f;
constexpr float cTopPanelAditionalButtonSize = cTabHeight - cTabYOffset;
constexpr float cSeparateBlocksSpacing = 12.0f;
constexpr float cSeparatorIndentMultiplier = 0.67f;

constexpr float cRibbonItemInterval = 4.0f;
constexpr float cRibbonItemMinWidth = 86.0f;
constexpr float cRibbonButtonWindowPaddingX = 6.0f;
constexpr float cRibbonButtonWindowPaddingY = 4.0f;
constexpr float cCheckboxPadding = 2.0f;
constexpr float cButtonPadding = 8.0f;
constexpr float cInputPadding = 9.0f;
constexpr float cDefaultItemSpacing = 8.0f;
constexpr float cDefaultInnerSpacing = 8.0f;
constexpr float cDefaultWindowPaddingX = 8.0f;
constexpr float cDefaultWindowPaddingY = 12.0f;
constexpr float cItemInfoIndent = 16.0f;

constexpr float cSmallItemDropSizeModifier = 0.5f;

constexpr float cHeaderQuickAccessFrameRounding = 3.0f;
constexpr float cHeaderQuickAccessXSpacing = 12.0f;
constexpr float cHeaderQuickAccessIconSize = 14.0f;
constexpr float cHeaderQuickAccessFrameSize = 24.0f;

constexpr float cMiddleIconSize = 15.0f;
constexpr float cSmallIconSize = 10.0f;
constexpr float cQuickAccessBarHeight = 40.0f;

constexpr float cScrollBarSize = 10.0f;
constexpr float cBigIconSize = 20.0f;

constexpr int cSmallFontSize = 11;
constexpr int cDefaultFontSize = 13;
constexpr int cBigFontSize = 15;
constexpr int cHeadlineFontSize = 20;

constexpr float cRadioButtonSize = 20.0f;

constexpr float cModalWindowWidth = 368.0f;
constexpr float cModalWindowPaddingX = 28.0f;
constexpr float cModalWindowPaddingY = 20.0f;
constexpr float cModalButtonWidth = 104.0f;

const float cRadioInnerSpacingX = 12.f;

namespace StyleConsts
{

constexpr Vector2f pluginItemSpacing{ 8, 10 };

namespace Modal
{

constexpr float bigTitlePadding = 22.0f;
constexpr float exitBtnSize = 24.0f;

}

namespace ProgressBar
{

constexpr float internalSpacing = 16.0f;
constexpr float rounding = 8.0f;

constexpr Color textColor = Color( 117, 125, 136 );

}

namespace CustomCombo
{

constexpr Vector2f framePadding{ 13, 8 };

} // CustomCombo

namespace Notification
{
constexpr float cWindowRounding = 4.f;
constexpr float cWindowSpacing = 20.f;
constexpr float cWindowBorderWidth = 2.f;
constexpr float cWindowPadding = 16.f;
constexpr float cNotificationWindowPaddingX = 10.f;
constexpr float cNotificationWindowPaddingY = 10.f;
constexpr float cWindowsPosY = 95.f;
constexpr float cHistoryButtonSizeY = 28.0f;
constexpr Vector2f cTextFramePadding{ 30, 8 };
constexpr float cTextFrameRounding = 8.0f;
} // Notification

} // MRStyle

}
