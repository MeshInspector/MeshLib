var postWasmLoad = function () {
    setTimeout(registerResize);
    setupColorTheme();
    updateCalculateMouseEvent();
    registerSafeDecoder();
}