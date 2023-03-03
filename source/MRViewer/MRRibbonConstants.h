#pragma once

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

constexpr float cMiddleIconSize = 16.0f;
constexpr float cSmallIconSize = 10.0f;
constexpr float cQuickAccessBarHeight = 40.0f;

constexpr float cScrollBarSize = 10.0f;
constexpr float cBigIconSize = 20.0f;

constexpr int cSmallFontSize = 11;
constexpr int cDefaultFontSize = 13;
constexpr int cBigFontSize = 15;
constexpr int cHeadlineFontSize = 24;

constexpr float cPaletteDiscretizationScaling = 5.0f / 18.0f;

constexpr float cRadioButtonSize = 20.0f;

constexpr float cModalWindowWidth = 368.0f;
constexpr float cModalWindowPaddingX = 28.0f;
constexpr float cModalWindowPaddingY = 20.0f;
constexpr float cModalButtonWidth = 104.0f;

struct ImGuiVec2
{
    float x{ 0 }, y{ 0 };
};

namespace MRStyle
{

constexpr ImGuiVec2 pluginItemSpacing{ 8, 10 };

namespace CustomCombo
{

constexpr ImGuiVec2 framePadding{ 13, 8 };

}; // CustomCombo

}; // MRStyle

}