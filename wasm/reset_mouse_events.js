var hasMouse = function () {
    return !(('ontouchstart' in window) || (navigator.maxTouchPoints > 0) || (navigator.msMaxTouchPoints > 0));
}
var lastTouches;
var storedTime = Date.now();
var fixControlTime = 0;
var zoomMode = false;

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

var getTouchesDistance = function (touches, rect) {
    if (touches === undefined || touches.length !== 2)
        return -1.0;

    var coord0 = getTouchPos(touches[0], rect);
    var coord1 = getTouchPos(touches[1], rect);

    var xDist = coord0.x - coord1.x;
    var yDist = coord0.y - coord1.y;

    return Math.sqrt(xDist * xDist + yDist * yDist);
}

var updateCalculateMouseEvent = function () {
    if (hasMouse())
        return;


    var getMouseButtonIdByEvent = function (event) {
        if (event.type === "touchend" || event.type === "touchcancel") {
            if (event.touches.length > 1)
                return -1;
            return event.touches.length > 0 ? 2 : 0;
        }
        if (event.touches.length > 2)
            return -1;
        return event.touches.length > 1 ? 2 : 0;
    };

    var oldCalcMovementFunction = Browser.calculateMouseEvent;

    Browser.calculateMouseEvent = function (event) {
        var rect = Module["canvas"].getBoundingClientRect();
        if (event.type === "touchstart" || event.type === "touchend" || event.type === "touchmove") {
            if (event.touches.length == 0) {
                return;
            }
            fixControlTime += Date.now() - storedTime;
            storedTime = Date.now();

            if (event.type === "touchstart") {
                fixControlTime = 0;
            }
            var touch = event.touches[0];
            event["button"] = getMouseButtonIdByEvent(event);
            var coords = getTouchPos(touch, rect);
            Browser.mouseX = coords.x;
            Browser.mouseY = coords.y;
            if (event.type === "touchstart") {
                Browser.lastTouches[touch.identifier] = coords;
                Browser.touches[touch.identifier] = coords
            } else if (event.type === "touchend" || event.type === "touchmove") {
                var last = Browser.touches[touch.identifier];
                if (!last)
                    last = coords;
                Browser.lastTouches[touch.identifier] = last;
                Browser.touches[touch.identifier] = coords
            }
            if (lastTouches !== undefined && lastTouches.length === 2 && event.touches.length === 2 &&
                event.type === "touchmove" && fixControlTime < 500) {
                var prevDist = getTouchesDistance(lastTouches, rect);
                var curDist = getTouchesDistance(event.touches, rect);
                if (prevDist > 0 && curDist > 0) {
                    var delta = curDist - prevDist;
                    if (zoomMode || Math.abs(delta) > 30) {
                        fixControlTime = 0;
                        zoomMode = true;
                        event.wheelDelta = -delta * 2;
                        var coords1 = getTouchPos(event.touches[1], rect);
                        Browser.mouseX = (Browser.mouseX + coords1.x) * 0.5;
                        Browser.mouseY = (Browser.mouseY + coords1.y) * 0.5;
                        GLFW.onMouseWheel(event);
                        event["button"] = -1;
                    }
                }
            }
            else {
                zoomMode = false;
            }
            lastTouches = event.touches;
            event.preventDefault();
            return
        }
        oldCalcMovementFunction(event);
        event.preventDefault();
        return;
    };

    Browser.getMouseWheelDelta = function (event) {
        var delta = 0;
        switch (event.type) {
            case "DOMMouseScroll":
                delta = event.detail / 3;
                break;
            case "mousewheel":
                delta = event.wheelDelta / 120;
                break;
            case "touchmove":
                delta = event.wheelDelta / 120;
                break;
            case "wheel":
                delta = event.deltaY;
                switch (event.deltaMode) {
                    case 0:
                        delta /= 100;
                        break;
                    case 1:
                        delta /= 3;
                        break;
                    case 2:
                        delta *= 80;
                        break;
                    default:
                        throw "unrecognized mouse wheel delta mode: " + event.deltaMode
                }
                break;
            default:
                throw "unrecognized mouse wheel event: " + event.type
        }
        return delta
    };
}
