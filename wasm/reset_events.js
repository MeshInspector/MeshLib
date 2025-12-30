var pointerCounter = 0;
var mouseState = [0, 0, 0];
var touchId = [-1, -1];
var reinterpretEvent = false;
var pointerSize = 0;

var overrideKeyDown = function (event) {
    GLFW.onKeydown(event);
    // suppress some browser hotkeys
    if (event.key == "F1" || event.key == "F2" || event.key == "F3" || event.key == "F4" || event.ctrlKey || event.metaKey)
        event.preventDefault();
}

var updateKeyEvents = function () {
    window.removeEventListener("keydown", GLFW.onKeydown, true);
    window.addEventListener("keydown", overrideKeyDown, true);
}

var keyboardEventsArePresent = true;

var removeKeyboardEvents = function () {
    if (keyboardEventsArePresent) {
        window.removeEventListener("keydown", overrideKeyDown, true);
        window.removeEventListener("keypress", GLFW.onKeyPress, true);
        window.removeEventListener("keyup", GLFW.onKeyup, true);
        keyboardEventsArePresent = false;
    }
};

var addKeyboardEvents = function () {
    if (!keyboardEventsArePresent) {
        window.addEventListener("keydown", overrideKeyDown, true);
        window.addEventListener("keypress", GLFW.onKeyPress, true);
        window.addEventListener("keyup", GLFW.onKeyup, true);
        keyboardEventsArePresent = true;
    }
    // enforce several frames to toggle animation when popup closed
    for (var i = 0; i < 500; i += 100)
        setTimeout(function () { Module.ccall('emsPostEmptyEvent', 'void', ['number'], [1]); }, i);
};

var getPointerSize = function () {
    if (!pointerSize) {
        pointerSize = Module.ccall('emsGetPointerSize', 'number', [], []);
    }
    return pointerSize;
}

var toPointer = function (value) {
    return getPointerSize() == 8 ? BigInt(value) : value;
}

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

var drop_files_or_dir = function (event) {
    if (!GLFW.active || !GLFW.active.dropFunc) return;
    if (!event.dataTransfer || !event.dataTransfer.files || event.dataTransfer.files.length == 0) return;
    event.preventDefault();

    var drop_dir = ".glfw_dropped_files";
    var filenames = _malloc(event.dataTransfer.files.length * getPointerSize());
    var filenamesArray = [];
    for (var i = 0; i < event.dataTransfer.files.length; ++i) {
        var path = "/" + drop_dir + "/" + event.dataTransfer.files[i].name.replace(/\//g, "_");
        var filename = stringToNewUTF8(path);
        filenamesArray.push(filename);
        if (getPointerSize() == 8)
            HEAPU64[(filenames + i * 8) / 8] = BigInt(filename);
        else if (typeof GROWABLE_HEAP_U32 !== 'undefined')
            GROWABLE_HEAP_U32()[filenames + i * 4 >> 2] = filename;
        else
            HEAPU32[filenames + i * 4 >> 2] = filename;
    }

    // Read and save the files to emscripten's FS
    var written = 0;
    FS.createPath("/", drop_dir);
    function save(file, in_path, numfiles) {
        var path = "/" + drop_dir + in_path + "/" + file.name.replace(/\//g, "_");
        var reader = new FileReader;
        reader.onloadend = e => {
            if (reader.readyState != 2) {
                // not DONE
                ++written;
                out("failed to read dropped file: " + file.name + ": " + reader.error);
                return;
            }
            var data = e.target.result;
            FS.writeFile(path, new Uint8Array(data));
            if (++written === numfiles) {
                if (typeof (dynCall_viii) == 'function')
                    dynCall_viii(GLFW.active.dropFunc, GLFW.active.id, filenamesArray.length, filenames);
                else if (typeof (dynCall_vjii) == 'function')
                    dynCall_vjii(GLFW.active.dropFunc, BigInt(GLFW.active.id), filenamesArray.length, filenames);
                else
                    getWasmTableEntry(GLFW.active.dropFunc)(toPointer(GLFW.active.id), filenamesArray.length, filenames);
                for (var i = 0; i < filenamesArray.length; ++i) {
                    _free(filenamesArray[i]);
                }
                _free(filenames);
            }
        };
        reader.readAsArrayBuffer(file);
    }

    let filesQ = [];
    var finalize = function () {
        count = filesQ.length;
        for (var i = 0; i < count; ++i)
            save(filesQ[i].file, filesQ[i].path, count);
    }

    if (typeof DataTransferItem.prototype.webkitGetAsEntry !== "undefined") {
        let entriesTree = {};
        var markDone = function (fullpath, recursive) {
            if (entriesTree[fullpath].subpaths.length != 0) return;
            delete entriesTree[fullpath];
            let parentpath = fullpath.substring(0, fullpath.lastIndexOf('/'));
            if (!entriesTree.hasOwnProperty(parentpath)) {
                if (Object.keys(entriesTree).length == 0) finalize();
                return;
            }
            const fpIndex = entriesTree[parentpath].subpaths.indexOf(fullpath);
            if (fpIndex > -1) entriesTree[parentpath].subpaths.splice(fpIndex, 1);
            if (recursive) markDone(parentpath, true);
            if (Object.keys(entriesTree).length == 0) finalize();
        };
        var processEntry = function (entry) {
            let fp = entry.fullPath;
            let pp = fp.substring(0, fp.lastIndexOf('/'));
            entriesTree[fp] = { subpaths: [] };
            if (entry.isFile) {
                entry.file((f) => { filesQ.push({ file: f, path: pp }); markDone(fp, false); })
            } else if (entry.isDirectory) {
                if (entriesTree.hasOwnProperty(pp)) entriesTree[pp].subpaths.push(fp);
                FS.createPath("/" + drop_dir + pp, entry.name);
                var reader = entry.createReader();
                var rRead = function (dirEntries) {
                    if (dirEntries.length == 0) {
                        markDone(fp, true);
                        return;
                    }
                    for (const ent of dirEntries) processEntry(ent);
                    reader.readEntries(rRead);
                };
                reader.readEntries(rRead);
            }
        };
        for (const item of event.dataTransfer.items) {
            processEntry(item.webkitGetAsEntry());
        }
    }
    else {
        // fallback for browsers that does not support `webkitGetAsEntry`
        for (var i = 0; i < event.dataTransfer.files.length; ++i) {
            filesQ.push({ file: event.dataTransfer.files[i], path: "" });
        }
        finalize();
    }
    return false;
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
    Module["canvas"].removeEventListener("dragover", GLFW.onDragover, true);
    Module["canvas"].removeEventListener("drop", GLFW.onDrop, true);

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
        if (typeof (dynCall_viiii) == 'function')
            dynCall_viiii(GLFW.active.mouseButtonFunc, GLFW.active.id, eventButton, status, GLFW.getModBits(GLFW.active));
        else if (typeof (dynCall_vjiii) == 'function')
            dynCall_vjiii(GLFW.active.mouseButtonFunc, BigInt(GLFW.active.id), eventButton, status, GLFW.getModBits(GLFW.active));
        else
            getWasmTableEntry(GLFW.active.mouseButtonFunc)(toPointer(GLFW.active.id), eventButton, status, GLFW.getModBits(GLFW.active));
    }

    GLFW.onMousemove = function (event) {
        if (!GLFW.active)
            return;
        if (!Browser.calculateMouseEvent(event)) return;
        if (event.target != Module["canvas"] || !GLFW.active.cursorPosFunc)
            return;
        if (typeof (dynCall_vidd) == 'function')
            dynCall_vidd(GLFW.active.cursorPosFunc, GLFW.active.id, Browser.mouseX, Browser.mouseY);
        else if (typeof (dynCall_vjdd) == 'function')
            dynCall_vjdd(GLFW.active.cursorPosFunc, BigInt(GLFW.active.id), Browser.mouseX, Browser.mouseY);
        else
            getWasmTableEntry(GLFW.active.cursorPosFunc)(toPointer(GLFW.active.id), Browser.mouseX, Browser.mouseY);
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
        if (typeof (dynCall_vidd) == 'function')
            dynCall_vidd(GLFW.active.scrollFunc, GLFW.active.id, sx, sy)
        else if (typeof (dynCall_vjdd) == 'function')
            dynCall_vjdd(GLFW.active.scrollFunc, BigInt(GLFW.active.id), sx, sy)
        else
            getWasmTableEntry(GLFW.active.scrollFunc)(toPointer(GLFW.active.id), sx, sy);
        preventFunc(event)
    }

    GLFW.onDragover = function (event) {
        if (!GLFW.active)
            return;
        Browser.setMouseCoords(event.pageX, event.pageY);
        Module.ccall('emsDragOver', 'void', ['number', 'number'], [Browser.mouseX, Browser.mouseY]);
        event.preventDefault();
        return false
    }

    GLFW.onDrop = drop_files_or_dir;

    // add new events
    Module["canvas"].addEventListener("pointermove", GLFW.onMousemove, true);
    Module["canvas"].addEventListener("pointerdown", GLFW.onMouseButtonDown, true);
    Module["canvas"].addEventListener("pointercancel", GLFW.onMouseButtonUp, true);
    Module["canvas"].addEventListener("pointerup", GLFW.onMouseButtonUp, true);
    Module["canvas"].addEventListener("wheel", GLFW.onMouseWheel, true);
    Module["canvas"].addEventListener("mousewheel", GLFW.onMouseWheel, true);
    Module["canvas"].addEventListener("dragover", GLFW.onDragover, true);
    Module["canvas"].addEventListener("drop", GLFW.onDrop, true);
    Module["canvas"].addEventListener("dragenter", (e) => {
        Module.ccall('emsDragEnter', 'void', [], []);
    }, true);
    Module["canvas"].addEventListener("dragleave", (e) => {
        Module.ccall('emsDragLeave', 'void', [], []);
        // enforce several frames to toggle animation on drag left
        for (var i = 0; i < 500; i += 100)
            setTimeout(function () { Module.ccall('emsPostEmptyEvent', 'void', ['number'], [1]); }, i);
    }, true);

    addEventListener('blur', (e) => {
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
            if (typeof (dynCall_viii) == 'function')
                dynCall_viii(win.windowSizeFunc, win.id, width, height);
            else if (typeof (dynCall_vjii) == 'function')
                dynCall_vjii(win.windowSizeFunc, win.id, width, height);
            else
                getWasmTableEntry(win.windowSizeFunc)(win.id, width, height);
        }
    }
}

var updateCalculateMouseEvent = function () {
    updateEvents();
}
