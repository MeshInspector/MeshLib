var postWasmLoad = function () {
    setTimeout(registerResize);
    updateCalculateMouseEvent();
    registerErrorHandling();
    registerSafeDecoder();
};
