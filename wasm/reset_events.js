var pointerCounter = 0;
var mouseState = [0, 0, 0];
var touchId = [-1, -1];
var reinterpretEvent = false;

var hasMouse = function () {
    return !(('ontouchstart' in window) || (navigator.maxTouchPoints > 0) || (navigator.msMaxTouchPoints > 0));
}

var postEmptyEvent = function (timer, forceFramesNum) {
    setTimeout(function () { Module.ccall('emsPostEmptyEvent', 'void', ['number'], [forceFramesNum]); }, timer);
}

var getPos = function (event, rect) {
    var cw = Module["canvas"].width;
    var ch = Module["canvas"].height;

    var scrollX = typeof window.scrollX != "undefined" ? window.scrollX : window.pageXOffset;
    var scrollY = typeof window.scrollY != "undefined" ? window.scrollY : window.pageYOffset;

    var adjustedX = event.pageX - (scrollX + rect.left);
    var adjustedY = event.pageY - (scrollY + rect.top);
    adjustedX = adjustedX * (cw / rect.width);
    adjustedY = adjustedY * (ch / rect.height);

    var coords = {
        x: adjustedX,
        y: adjustedY
    };
    return coords;
}

// override this to prevent only events you need to prevent
var preventFunc = function (event) {
    event.preventDefault();
}

var updateEvents = function () {
    // remove all touch events
    Module["canvas"].removeEventListener("touchmove", GLFW.onMousemove, true);
    Module["canvas"].removeEventListener("touchstart", GLFW.onMouseButtonDown, true);
    Module["canvas"].removeEventListener("touchcancel", GLFW.onMouseButtonUp, true);
    Module["canvas"].removeEventListener("touchend", GLFW.onMouseButtonUp, true);

    // remove mouse events
    Module["canvas"].removeEventListener("mousemove", GLFW.onMousemove, true);
    Module["canvas"].removeEventListener("mousedown", GLFW.onMouseButtonDown, true);
    Module["canvas"].removeEventListener("mouseup", GLFW.onMouseButtonUp, true);
    Module["canvas"].removeEventListener("wheel", GLFW.onMouseWheel, true);
    Module["canvas"].removeEventListener("mousewheel", GLFW.onMouseWheel, true);

    // make own touch events callbacks
    var touchEventProcess = function (event, funcName) {
        var rect = Module["canvas"].getBoundingClientRect();
        var coords = getPos(event, rect);
        if (event.isPrimary && pointerCounter == 1) {
            Browser.mouseX = coords.x;
            Browser.mouseY = coords.y;
        }
        Module.ccall(funcName, 'number', ['number', 'number', 'number'],
            [event.pointerId, coords.x, coords.y]);
    }

    // main event func determine what to call
    var oldCalcMovementFunction = Browser.calculateMouseEvent;

    var touchDownFunc = function (event) {
        var proceed = true;
        if (touchId[0] == -1)
            touchId[0] = event.pointerId;
        else if (touchId[1] == -1)
            touchId[1] = event.pointerId;
        else
            proceed = false;
        if (proceed) {
            pointerCounter++;
            touchEventProcess(event, 'emsTouchStart');
        }
    }

    Browser.calculateMouseEvent = function (event) {
        var bubbleUp = true;
        if (event.pointerType == "mouse") {
            var eventButton = GLFW.DOMToGLFWMouseButton(event);
            if (event.type == "pointermove" && !reinterpretEvent) {
                if (eventButton >= 0) {
                    reinterpretEvent = true;
                    if (mouseState[eventButton] == 0 && event.buttons != 0)
                        mouseState[eventButton] = 1;
                    else
                        mouseState[eventButton] = 0;
                    GLFW.onMouseButtonChanged(event, mouseState[eventButton]);
                    return false;
                }
            }
            if (event.type == "pointerdown")
                mouseState[eventButton] = 1;
            else if (event.type == "pointerup" || event.type == "pointercancel")
                mouseState[eventButton] = 0;
            reinterpretEvent = false;
            oldCalcMovementFunction(event);
        }
        else if (event.pointerType == "touch") {
            if (event.type == "pointerdown") {
                touchDownFunc(event);
            }
            else if (event.type == "pointermove") {
                if (touchId[0] == event.pointerId || touchId[1] == event.pointerId)
                    touchEventProcess(event, 'emsTouchMove');
                else
                    touchDownFunc(event);
            }
            else if (event.type == "pointerup" || event.type == "pointercancel") {
                var proceed = true;
                if (touchId[0] == event.pointerId)
                    touchId[0] = -1;
                else if (touchId[1] == event.pointerId)
                    touchId[1] = -1;
                else
                    proceed = false;
                if (proceed) {
                    touchEventProcess(event, 'emsTouchEnd');
                    pointerCounter--;
                }
            }
            bubbleUp = false;
        }
        return bubbleUp;
    };

    GLFW.onMouseButtonChanged = function (event, status) {
        if (!GLFW.active) return;
        if (!Browser.calculateMouseEvent(event)) return;
        if (event.target != Module["canvas"]) return;
        var eventButton = GLFW.DOMToGLFWMouseButton(event);
        if (status == 1) {
            GLFW.active.buttons |= 1 << eventButton;
            event.target.setPointerCapture(event.pointerId);
        } else {
            GLFW.active.buttons &= ~(1 << eventButton);
        }
        if (!GLFW.active.mouseButtonFunc) return;
        if (typeof (dynCall_vidd) == 'undefined')
            getWasmTableEntry(GLFW.active.mouseButtonFunc)(GLFW.active.id, eventButton, status, GLFW.getModBits(GLFW.active));
        else
            (function (a1, a2, a3, a4) {
                dynCall_viiii.apply(null, [GLFW.active.mouseButtonFunc, a1, a2, a3, a4]);
            })(GLFW.active.id, eventButton, status, GLFW.getModBits(GLFW.active));
    }

    GLFW.onMousemove = function (event) {
        if (!GLFW.active)
            return;
        if (!Browser.calculateMouseEvent(event)) return;
        if (event.target != Module["canvas"] || !GLFW.active.cursorPosFunc)
            return;
        if (typeof (dynCall_vidd) == 'undefined')
            getWasmTableEntry(GLFW.active.cursorPosFunc)(GLFW.active.id, Browser.mouseX, Browser.mouseY);
        else
            (function (a1, a2, a3) {
                dynCall_vidd.apply(null, [GLFW.active.cursorPosFunc, a1, a2, a3]);
            })(GLFW.active.id, Browser.mouseX, Browser.mouseY);
    }

    GLFW.onMouseWheel = function (event) {
        // This is copypasted from Emscripten-generated JS code,
        // except we changed `event.preventDefault()` to `preventFunc(event)`.
        var delta = -Browser.getMouseWheelDelta(event);
        delta = delta == 0 ? 0 :
            delta > 0 ? Math.max(delta, 1) :
                Math.min(delta, -1);
        GLFW.wheelPos += delta;
        if (!GLFW.active || !GLFW.active.scrollFunc ||
            event.target != Module['canvas'])
            return;
        var sx = 0;
        var sy = delta;
        if (event.type == 'mousewheel') {
            sx = event.wheelDeltaX
        } else {
            sx = event.deltaX
        }
        if (typeof (dynCall_vidd) == 'undefined')
            getWasmTableEntry(GLFW.active.scrollFunc)(GLFW.active.id, sx, sy);
        else
            (function (a1, a2, a3) {
                dynCall_vidd.apply(null, [GLFW.active.scrollFunc, a1, a2, a3])
            })(GLFW.active.id, sx, sy);
        preventFunc(event)
    }

    // add new events
    Module["canvas"].addEventListener("pointermove", GLFW.onMousemove, true);
    Module["canvas"].addEventListener("pointerdown", GLFW.onMouseButtonDown, true);
    Module["canvas"].addEventListener("pointercancel", GLFW.onMouseButtonUp, true);
    Module["canvas"].addEventListener("pointerup", GLFW.onMouseButtonUp, true);
    Module["canvas"].addEventListener("wheel", GLFW.onMouseWheel, true);
    Module["canvas"].addEventListener("mousewheel", GLFW.onMouseWheel, true);
    addEventListener('blur', (event) => {
        Module.ccall('emsDropEvents', 'void', [], []);
    });
    // prevent others
    Module["canvas"].addEventListener("touchmove", preventFunc, true);
    Module["canvas"].addEventListener("touchstart", preventFunc, true);
    Module["canvas"].addEventListener("touchcancel", preventFunc, true);
    Module["canvas"].addEventListener("touchend", preventFunc, true);

    Module["canvas"].addEventListener("mousemove", preventFunc, true);
    Module["canvas"].addEventListener("mousedown", preventFunc, true);
    Module["canvas"].addEventListener("mouseup", preventFunc, true);

    GLFW.setWindowSize = function (winid, width, height) {
        var win = GLFW.WindowFromId(winid);
        if (!win) return;
        if (GLFW.active.id == win.id) {
            Browser.exitFullscreen();
            Browser.setCanvasSize(width, height);
            win.width = width;
            win.height = height;
        }
        if (win.windowSizeFunc) {
            if (typeof (dynCall_viii) == 'undefined')
                getWasmTableEntry(win.windowSizeFunc)(win.id, width, height);
            else
                ((a1, a2, a3) => dynCall_viii.apply(null, [win.windowSizeFunc, a1, a2, a3]))(win.id, width, height);
        }
    }
}

var updateCalculateMouseEvent = function () {
    updateEvents();
}
