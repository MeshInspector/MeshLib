var web_req_headers = [];
var web_req_params = [];
var web_req_body = "";
var web_req_timeout = 10000;
var web_req_method = 0;

var web_req_add_header = function (key, value) {
    web_req_headers.push({ key, value });
}

var web_req_add_param = function (key, value) {
    web_req_params.push({ key, value });
}

var web_req_clear = function () {
    web_req_headers = [];
    web_req_params = [];
    web_req_body = "";
    web_req_timeout = 10000;
    web_req_method = 0;
}

var web_req_send = function (url, async) {
    var method;
    if (web_req_method == 0)
        method = "GET";
    else
        method = "POST";

    for (var i = 0; i < web_req_params.length; i++) {
        if (i == 0)
            url += "?";
        else
            url += "&";
        url += web_req_params[i].key + "=" + web_req_params[i].value;
    }
    var req = new XMLHttpRequest();
    if (async)
        req.timeout = web_req_timeout;
    req.open(method, url, async);
    for (var i = 0; i < web_req_headers.length; i++) {
        req.setRequestHeader(web_req_headers[i].key, web_req_headers[i].value);
    }
    req.onloadend = (e) => {
        var res = {
            code: req.status,
            text: req.responseText,
            error: req.statusText
        };
        Module.ccall('emsCallResponseCallback', 'number', ['string'], [JSON.stringify(res)]);
    };
    req.send(web_req_body);
}
