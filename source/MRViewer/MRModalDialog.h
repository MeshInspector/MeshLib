#pragma once

#include "exports.h"

#include <functional>
#include <string>

namespace MR
{

/// \brief Settings for ModalDialog.
struct ModalDialogSettings
{
    /// Dialog window width. If the value is zero or negative, defaults to cModalWindowWidth * menuScaling.
    float windowWidth = 0.f;
    /// Render a centered headline text at the beginning of the dialog. The RibbonFontManager::FontType::Headline font
    /// is used if exists.
    std::string headline;
    /// Add a close button next in the top right corner of the dialog.
    bool closeButton = false;
    /// Render a centered text under the headline.
    std::string text;
    /// If set, add a "Don't show the dialog again" checkbox at the end of the dialog. The checkbox value is bound to
    /// the field value.
    bool* dontShowAgain = nullptr;
    /// If set, close the dialog on mouse click outside of it.
    bool closeOnClickOutside = false;
    /// Callback for the window close event.
    std::function<void ()> onWindowClose;
};

/// \brief Helper class to display modal dialogs.
/// \ref ModalDialogSettings
class ModalDialog
{
public:
    MRVIEWER_API ModalDialog( std::string label, ModalDialogSettings settings );

    /// Open the dialog and render its header (headline, close button, text).
    /// Returns true if the dialog window is open.
    MRVIEWER_API bool beginPopup( float menuScaling );
    /// Render the dialog's footer ("Don't show again" checkbox) and finish the dialog.
    MRVIEWER_API void endPopup( float menuScaling );

    /// Returns the current window width in pixels.
    [[nodiscard]] MRVIEWER_API static float windowWidth();

private:
    std::string label_;
    ModalDialogSettings settings_;
};

} // namespace MR
