var unexpectedErrorIsNotEnoughMemoryError = false;

var notEnoughMemoryError = function () {
    unexpectedErrorIsNotEnoughMemoryError = true;
};

var isUnexpectedErrorOccured = false;

var unexpectedError = function () {
    if (isUnexpectedErrorOccured)
        return;
    isUnexpectedErrorOccured = true;

    // Remove event listeners introduced in resize.js otherwise we will see canvas again on resize
    window.removeEventListener('resize', resizeCallBack);
    window.removeEventListener('orientationchange', resizeCallBack);
    document.getElementById('canvas').setAttribute('style', 'visibility: hidden;');
    showUnexpectedErrorPopup(unexpectedErrorPopupSettings());
};

var unexpectedErrorPopupSettings = function () {
    var reloadButton = document.createElement('div');
    reloadButton.setAttribute('onclick', 'location.reload()');
    reloadButton.innerHTML = 'Reload';

    return {
        width: 530,
        height: 180,
        title: unexpectedErrorIsNotEnoughMemoryError ? 'Not Enough Memory' : 'Unexpected Error Occurred',
        messageText: 'Try to Reload the Page',
        buttonElements: [reloadButton]
    };
};

var showUnexpectedErrorPopup = function (settings) {
    var { overlay, popup } = createOverlayPopup('show_unexpected_error_dialog', settings.title, settings.width, settings.height, true, false);

    var content = document.createElement('div');
    content.setAttribute('style', 'margin: 50px 20px 20px 20px;display: flex;flex-direction: column;align-items: center;gap: 20px;');

    var label = document.createElement('label');
    label.setAttribute('style', 'font-size: 14px;color: #fff;');
    label.innerHTML = settings.messageText;
    label.setAttribute('class', 'unselectable');

    var buttons = document.createElement('div');
    buttons.setAttribute('style', 'display: flex;gap: 10px');

    for (const buttonElement of settings.buttonElements) {
        buttonElement.setAttribute('style', 'text-decoration: none;width: 200px;height: 28px;border-radius: 4px;color: #fff;font-size: 14px;font-weight: 600;border: none;display: flex;align-items: center;justify-content: center;');
        buttonElement.setAttribute('class', 'button unselectable');
        buttons.appendChild(buttonElement);
    }

    content.appendChild(label);
    content.appendChild(buttons);

    popup.appendChild(content);

    removeKeyboardEvents();
    document.body.appendChild(overlay);
};

var registerErrorHandling = function () {
    quit_ = function (status, toThrow) {
        unexpectedError();
        throw toThrow;
    };
    window.onerror = (e) => {
        let errorStr = JSON.stringify(e);
        if (errorStr.includes("RuntimeError: Aborted()"))
            quit_("RuntimeError: Aborted()", e);
    };
};
