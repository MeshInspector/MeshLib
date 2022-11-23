var pointerCounter = 0;

var hasMouse = function () {
    return !(('ontouchstart' in window) || (navigator.maxTouchPoints > 0) || (navigator.msMaxTouchPoints > 0));
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

    Browser.calculateMouseEvent = function (event) {
        var bubbleUp = true;
        if (event.pointerType == "mouse")
            oldCalcMovementFunction(event);
        else if (event.pointerType == "touch") {
            if (event.type == "pointerdown")
            {
                pointerCounter++;
                touchEventProcess(event, 'emsTouchStart');
            }
            else if (event.type == "pointermove")
            {
                touchEventProcess(event, 'emsTouchMove');
            }
            else if (event.type == "pointerup" || event.type == "pointercancel")
            {
                touchEventProcess(event, 'emsTouchEnd');
                pointerCounter--;
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
        getWasmTableEntry(GLFW.active.mouseButtonFunc)(GLFW.active.id, eventButton, status, GLFW.getModBits(GLFW.active));
    }

    GLFW.onMousemove = function (event) {
        if (!GLFW.active)
            return;
        if (!Browser.calculateMouseEvent(event)) return;
        if (event.target != Module["canvas"] || !GLFW.active.cursorPosFunc)
            return;
        getWasmTableEntry(GLFW.active.cursorPosFunc)(GLFW.active.id, Browser.mouseX, Browser.mouseY)
    }

    // add new events
    Module["canvas"].addEventListener("pointermove", GLFW.onMousemove, true);
    Module["canvas"].addEventListener("pointerdown", GLFW.onMouseButtonDown, true);
    Module["canvas"].addEventListener("pointercancel", GLFW.onMouseButtonUp, true);
    Module["canvas"].addEventListener("pointerup", GLFW.onMouseButtonUp, true);
    // prevent others
    Module["canvas"].addEventListener("touchmove", preventFunc, true);
    Module["canvas"].addEventListener("touchstart", preventFunc, true);
    Module["canvas"].addEventListener("touchcancel", preventFunc, true);
    Module["canvas"].addEventListener("touchend", preventFunc, true);

    Module["canvas"].addEventListener("mousemove", preventFunc, true);
    Module["canvas"].addEventListener("mousedown", preventFunc, true);
    Module["canvas"].addEventListener("mouseup", preventFunc, true);

}

var updateCalculateMouseEvent = function () {
    updateEvents();
}
