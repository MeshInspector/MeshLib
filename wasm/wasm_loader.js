var setWasmLoadProgress = function (proc) {
    progressElement = document.getElementById("progress");
    progressHolder = document.getElementById("progress-holder");
    progressHolder.setAttribute('style', 'visibility: visible');
    progressElement.setAttribute('style', 'width:' + proc + "%");
}

var clearWasmLoadProgress = function () {
    progressHolder = document.getElementById("progress-holder");
    progressHolder.setAttribute('style', 'visibility: hidden');
}

var statusElement = document.getElementById("status")
    , logoElement = document.getElementById("logo")
    , spinnerElement = document.getElementById("spinner")
    , Module = {
        preRun: [],
        postRun: [],
        print: function () {
            var e = document.getElementById("output");
            return e && (e.value = ""),
                function (t) {
                    arguments.length > 1 && (t = Array.prototype.slice.call(arguments).join(" ")),
                        console.log(t),
                        e && (e.value += t + "\n",
                            e.scrollTop = e.scrollHeight)
                }
        }(),
        canvas: function () {
            var e = document.getElementById("canvas");
            return e.addEventListener("webglcontextlost", (function (e) {
                unexpectedError(),
                    e.preventDefault()
            }
            ), !1),
                e
        }(),
        setStatus: function (e) {
            if (Module.setStatus.last || (Module.setStatus.last = {
                time: Date.now(),
                text: ""
            }),
                e !== Module.setStatus.last.text) {
                var t = e.match(/([^(]+)\((\d+(\.\d+)?)\/(\d+)\)/)
                    , n = Date.now();
                t && n - Module.setStatus.last.time < 30 || (Module.setStatus.last.time = n,
                    Module.setStatus.last.text = e,
                    t ? (e = t[1],
                        setWasmLoadProgress(100 * parseInt(t[2]) / parseInt(t[4])),
                        spinnerElement.hidden = !1) : (
                            e || (spinnerElement.style.display = "none",
                                statusElement.style.display = "none",
                                logoElement.style.display = "none",
                                clearWasmLoadProgress(),
                                postWasmLoad()
                            )),
                    statusElement.innerHTML = e)
            }
        },
        totalDependencies: 0,
        monitorRunDependencies: function (e) {
            this.totalDependencies = Math.max(this.totalDependencies, e),
                Module.setStatus(e ? "Preparing... (" + (this.totalDependencies - e) + "/" + this.totalDependencies + ")" : "All downloads complete.")
        }
    };
Module.setStatus("Downloading..."),
    window.onerror = function (e) {
        Module.setStatus("Exception thrown, see JavaScript console"),
            spinnerElement.style.display = "none",
            statusElement.style.display = "none",
            Module.setStatus = function (e) {
                e && Module.printErr("[post-exception status] " + e)
            }
    }
