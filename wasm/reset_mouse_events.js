var hasMouse = function () {
    return !(('ontouchstart' in window) || (navigator.maxTouchPoints > 0) || (navigator.msMaxTouchPoints > 0));
}

var lastTouchEvenTime = 0;

var getTouchPos = function (touch, rect) {
    var cw = Module["canvas"].width;
    var ch = Module["canvas"].height;

    var scrollX = typeof window.scrollX != "undefined" ? window.scrollX : window.pageXOffset;
    var scrollY = typeof window.scrollY != "undefined" ? window.scrollY : window.pageYOffset;

    var adjustedX = touch.pageX - (scrollX + rect.left);
    var adjustedY = touch.pageY - (scrollY + rect.top);
    adjustedX = adjustedX * (cw / rect.width);
    adjustedY = adjustedY * (ch / rect.height);

    var coords = {
        x: adjustedX,
        y: adjustedY
    };
    return coords;
}

var updateBaseEventsFunctions = function () {
    // remove all touch events
    Module["canvas"].removeEventListener("touchmove", GLFW.onMousemove, true);
    Module["canvas"].removeEventListener("touchstart", GLFW.onMouseButtonDown, true);
    Module["canvas"].removeEventListener("touchcancel", GLFW.onMouseButtonUp, true);
    Module["canvas"].removeEventListener("touchend", GLFW.onMouseButtonUp, true);
    // change mouse events to pointer events
    if (hasMouse()) {
        Module["canvas"].removeEventListener("mousedown", GLFW.onMouseButtonDown, true);
        Module["canvas"].removeEventListener("mouseup", GLFW.onMouseButtonUp, true);
        Module["canvas"].addEventListener("pointerdown", GLFW.onMouseButtonDown, true);
        Module["canvas"].addEventListener("pointerup", GLFW.onMouseButtonUp, true);
    }

    var oldCalcMovementFunction = Browser.calculateMouseEvent;

    Browser.calculateMouseEvent = function (event) {
        console.log(event)
        event.preventDefault();
        if (Date.now() - lastTouchEvenTime < 200)
            return false;
        oldCalcMovementFunction(event);
        return true;
    };

    // support pointer capture
    GLFW.onMouseButtonChanged = function (event, status) {
        if (!GLFW.active) return;
        if (!Browser.calculateMouseEvent(event)) return;
        if (event.target != Module["canvas"]) return;
        var eventButton = GLFW.DOMToGLFWMouseButton(event);
        if (status == 1) {
            GLFW.active.buttons |= 1 << eventButton;
            if (hasMouse())
                event.target.setPointerCapture(event.pointerId);
        } else {
            GLFW.active.buttons &= ~(1 << eventButton);
        }
        if (!GLFW.active.mouseButtonFunc) return;
        getWasmTableEntry(GLFW.active.mouseButtonFunc)(GLFW.active.id, eventButton, status, GLFW.getModBits(GLFW.active));
    }

    var touchEventProcess = function (event, funcName) {
        var rect = Module["canvas"].getBoundingClientRect();
        lastTouchEvenTime = Date.now();
        var res = 0;
        for (let i = 0; i < event.changedTouches.length; i++) {
            var coords = getTouchPos(event.changedTouches[i], rect);
            if (event.changedTouches[i].identifier == 0) {
                Browser.mouseX = coords.x;
                Browser.mouseY = coords.y;
            }
            res += Module.ccall(funcName, 'number', ['number', 'number', 'number'],
                [event.changedTouches[i].identifier, coords.x, coords.y]);
        }
        //if (res != 0) {
        event.preventDefault();
        //}
    }

    var touchStartEvent = function (event) {
        touchEventProcess(event, 'emsTouchStart');
    }

    var touchMoveEvent = function (event) {
        touchEventProcess(event, 'emsTouchMove');
    }

    var touchEndEvent = function (event) {
        touchEventProcess(event, 'emsTouchEnd');
    }

    // add new touch events    
    Module["canvas"].addEventListener("touchmove", touchMoveEvent, true);
    Module["canvas"].addEventListener("touchstart", touchStartEvent, true);
    Module["canvas"].addEventListener("touchcancel", touchEndEvent, true);
    Module["canvas"].addEventListener("touchend", touchEndEvent, true);
}

var updateCalculateMouseEvent = function () {
    updateBaseEventsFunctions();
}
