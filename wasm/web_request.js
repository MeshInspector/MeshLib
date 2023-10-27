var web_req_headers = [];
var web_req_params = [];
var web_req_body = "";
var web_req_formdata = null;
var web_req_timeout = 10000;
var web_req_method = 0;
var web_req_output_path = "";

var web_req_add_header = function (key, value) {
    web_req_headers.push({ key, value });
}

var web_req_add_param = function (key, value) {
    web_req_params.push({ key, value });
}

var web_req_add_formdata = function (path, contentType, name, fileName) {
    if (!FS.analyzePath(path).exists)
        return;
    const content = FS.readFile(path);

    if (web_req_formdata == null)
        web_req_formdata = new FormData();
    web_req_formdata.append(name, new Blob(content, {type: contentType}), fileName);
}

var web_req_clear = function () {
    web_req_headers = [];
    web_req_params = [];
    web_req_body = "";
    web_req_formdata = null;
    web_req_timeout = 10000;
    web_req_method = 0;
    web_req_output_path = "";
}

var web_req_send = function (url, async, ctxId) {
    var urlCpy = url;
    for (var i = 0; i < web_req_params.length; i++) {
        if (i == 0)
            url += "?";
        else
            url += "&";
        url += web_req_params[i].key + "=" + web_req_params[i].value;
    }

    var method;
    if (web_req_method == 0)
        method = "GET";
    else
        method = "POST";

    var headers = new Headers();
    for (var i = 0; i < web_req_headers.length; i++) {
        headers.append(web_req_headers[i].key, web_req_headers[i].value);
    }

    var outputPath = web_req_output_path;

    var options = {
        method: method,
        headers: headers,
    };
    if (web_req_method != 0) {
        options.body = web_req_formdata ?? web_req_body;
    }
    if (async) {
        const controller = new AbortController();
        setTimeout(() => controller.abort(), web_req_timeout);
        options.signal = controller.signal;
    }

    fetch(url, options).then(async (response) => {
        var res = {
            url: urlCpy,
            code: response.status,
            text: "",
            error: response.statusText,
        };
        if (response.ok) {
            if (outputPath) {
                FS.writeFile(outputPath, new Uint8Array(await response.arrayBuffer()));
            } else {
                res.text = await response.text();
            }
        }
        Module.ccall('emsCallResponseCallback', 'number', ['string', 'bool', 'number'], [JSON.stringify(res), async, ctxId]);
    });
}
