var postWasmLoad = function () {
    setTimeout(registerResize);
    //setupColorTheme();
    updateCalculateMouseEvent();
    updateKeyEvents();
    registerErrorHandling();
};
