class WebReqCtx {
    headers = [];
    params = [];
    body = "";
    formdata = null;
    filename = "";
    timeout = 10000;
    method = "GET";
    use_upload_callback = false;
    use_download_callback = false;
}

var web_req_ctxs = {};

var create_web_ctx_if_needed = function (ctxId) {
    if (ctxId in web_req_ctxs)
        return;
    web_req_ctxs[ctxId] = new WebReqCtx();
}

var web_req_add_header = function (key, value, ctxId) {
    create_web_ctx_if_needed(ctxId)
    web_req_ctxs[ctxId].headers.push({ key, value });
}

var web_req_add_param = function (key, value, ctxId) {
    create_web_ctx_if_needed(ctxId);
    web_req_ctxs[ctxId].params.push({ key, value });
}

var web_req_add_formdata = function (path, contentType, name, fileName, ctxId) {
    if (!FS.analyzePath(path).exists)
        return;
    const content = FS.readFile(path);
    create_web_ctx_if_needed(ctxId);
    if (web_req_ctxs[ctxId].formdata == null)
        web_req_ctxs[ctxId].formdata = new FormData();
    web_req_ctxs[ctxId].formdata.append(name, new Blob([content], { type: contentType }), fileName);
}

var web_req_clear = function (ctxId) {
    if (ctxId in web_req_ctxs)
        delete web_req_ctxs[ctxId];
}

var web_req_send = function (url, async, ctxId) {
    create_web_ctx_if_needed(ctxId);
    var urlCpy = url;
    for (var i = 0; i < web_req_ctxs[ctxId].params.length; i++) {
        if (i == 0)
            url += "?";
        else
            url += "&";
        url += web_req_ctxs[ctxId].params[i].key + "=" + web_req_ctxs[ctxId].params[i].value;
    }
    var req = new XMLHttpRequest();
    if (async)
        req.timeout = web_req_ctxs[ctxId].timeout;
    req.open(web_req_ctxs[ctxId].method, url, async);
    for (var i = 0; i < web_req_ctxs[ctxId].headers.length; i++) {
        req.setRequestHeader(web_req_ctxs[ctxId].headers[i].key, web_req_ctxs[ctxId].headers[i].value);
    }
    if (web_req_ctxs[ctxId].use_upload_callback) {
        req.upload.addEventListener('progress', (event) => {
            if (event.lengthComputable) {
                var v = event.loaded / event.total;
                Module.ccall('emsCallUploadCallback', 'number', ['number', 'number'], [v, ctxId]);
            }
        });
    }
    if (web_req_ctxs[ctxId].use_download_callback) {
        req.addEventListener('progress', (event) => {
            if (event.lengthComputable) {
                var v = event.loaded / event.total;
                Module.ccall('emsCallDownloadCallback', 'number', ['number', 'number'], [v, ctxId]);
            }
        });
    }
    req.onloadend = (e) => {
        var res = {
            url: urlCpy,
            code: req.status,
            text: req.responseText,
            error: req.statusText
        };
        Module.ccall('emsCallResponseCallback', 'number', ['string', 'bool', 'number'], [JSON.stringify(res), async, ctxId]);
    };
    var payload = null;
    if (web_req_ctxs[ctxId].filename != "") {
        if (FS.analyzePath(web_req_ctxs[ctxId].filename).exists) {
            const content = FS.readFile(web_req_ctxs[ctxId].filename);
            payload = new Blob([content]);
        }
    }
    req.send(payload ?? web_req_ctxs[ctxId].formdata ?? web_req_ctxs[ctxId].body);
}

var web_req_async_download = function (url, outputPath, ctxId) {
    create_web_ctx_if_needed(ctxId);
    var urlCpy = url;
    for (var i = 0; i < web_req_ctxs[ctxId].params.length; i++) {
        if (i == 0)
            url += "?";
        else
            url += "&";
        url += web_req_ctxs[ctxId].params[i].key + "=" + web_req_ctxs[ctxId].params[i].value;
    }

    var headers = new Headers();
    for (var i = 0; i < web_req_ctxs[ctxId].headers.length; i++) {
        headers.append(web_req_ctxs[ctxId].headers[i].key, web_req_ctxs[ctxId].headers[i].value);
    }

    var payload = null;
    if (web_req_ctxs[ctxId].filename != "") {
        if (FS.analyzePath(web_req_ctxs[ctxId].filename).exists) {
            const content = FS.readFile(web_req_ctxs[ctxId].filename);
            payload = new Blob([content]);
        }
    }

    var options = {
        method: web_req_ctxs[ctxId].method,
        headers: headers,
    };
    if (web_req_ctxs[ctxId].method != "GET" && web_req_ctxs[ctxId].method != "HEAD") {
        options.body = payload ?? web_req_ctxs[ctxId].formdata ?? web_req_ctxs[ctxId].ody;
    }

    const controller = new AbortController();
    setTimeout(() => controller.abort(), web_req_ctxs[ctxId].timeout);
    options.signal = controller.signal;

    // TODO: upload progress callback support for Fetch API
    var use_download_callback = web_req_ctxs[ctxId].use_download_callback;

    fetch(url, options)
        .then(async (response) => {
            if (!use_download_callback) {
                return response;
            }
            if (!response.ok || !response.body) {
                // nothing to process
                return response;
            }
            const contentEncoding = response.headers.get('content-encoding');
            const contentLength = response.headers.get(contentEncoding ? 'x-file-size' : 'content-length');
            if (contentLength === null) {
                return response;
            }
            const total = parseInt(contentLength, 10);
            if (total == 0) {
                return response;
            }

            let loaded = 0;
            return new Response(
                new ReadableStream({
                    async start(controller) {
                        const reader = response.body.getReader();
                        for (; ;) {
                            const { done, value } = await reader.read();
                            if (done) {
                                break;
                            }
                            loaded += value.byteLength;
                            var v = loaded / total;
                            Module.ccall('emsCallDownloadCallback', 'number', ['number', 'number'], [v, ctxId]);
                            controller.enqueue(value);
                        }
                        controller.close();
                    }
                })
            );
        })
        .then(async (response) => {
            var res = {
                url: urlCpy,
                code: response.status,
                text: "",
                error: response.statusText,
            };
            if (response.ok) {
                FS.writeFile(outputPath, new Uint8Array(await response.arrayBuffer()));
            }
            Module.ccall('emsCallResponseCallback', 'number', ['string', 'bool', 'number'], [JSON.stringify(res), true, ctxId]);
        });
}
