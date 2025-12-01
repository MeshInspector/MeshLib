var ComposeHTMLHead = function (styles, images) {
    for (let i = 0; i < styles.length; ++i) {
        var link = document.createElement("link");
        link.setAttribute("rel", "stylesheet");
        link.setAttribute("href", styles[i] + window.location.search);
        document.head.appendChild(link);
    }
    for (let i = 0; i < images.length; ++i) {
        var link = document.createElement("link");
        link.setAttribute("rel", "preload");
        link.setAttribute("as", "image");
        link.setAttribute("href", images[i] + window.location.search);
        document.head.appendChild(link);
    }
}

var ComposeHTMLBody = function (scripts, deferScripts) {
    for (let i = 0; i < scripts.length; ++i) {
        var scr = document.createElement("script");
        scr.setAttribute("src", scripts[i] + window.location.search);
        scr.async = false; // dynamic scripts are considered async, so explicitly disable it
        document.body.appendChild(scr);
    }

    for (let i = 0; i < deferScripts.length; ++i) {
        var scr = document.createElement("script");
        scr.setAttribute("src", deferScripts[i] + window.location.search);
        scr.setAttribute("defer", "");
        scr.async = false; // dynamic scripts are considered async, so explicitly disable it
        document.body.appendChild(scr);
    }
}

ComposeHTMLHead(["styles.css"], ["tool_not_supp.svg"]);
ComposeHTMLBody([
    "ios_detect.js",
    "color_theme.js",
    "reset_events.js",
    "popups.js",
    "io_files.js",
    "open_link.js",
    "resize.js",
    "setter_cursor_type.js",
    "web_request.js",
    "version.js",
    "post_load.js",
    "wasm_loader.js",
    "config.js",
    "error.js",
    "download_desktop_window.js"
], ["MeshViewer.js"]);