var postWasmLoad = function () {
    setTimeout(registerResize);
    //setupColorTheme();
    updateCalculateMouseEvent();
    registerErrorHandling();
    registerSafeDecoder();
};
