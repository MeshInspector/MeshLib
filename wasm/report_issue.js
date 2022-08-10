var report_issue = function (repo, title, description, token) {
    var method = "POST";
    var url = "https://api.github.com/repos/MeshInspector/" + repo + "/issues";
    var acync = false;

    var body = { "title": title, "body": description };

    var req = new XMLHttpRequest();
    req.open(method, url, acync);

    req.setRequestHeader("authorization", "token " + token);
    req.setRequestHeader("Accept", "application/vnd.github.v3+json");
    req.send(JSON.stringify(body));
    var reqObj = JSON.parse(req.responseText);
    var text = req.status == 201 ? "OK" : reqObj["message"];
    var link = req.status == 201 ? reqObj["html_url"] : "None";
    Module.ccall('ems_report_issue_response', 'number', ['string', 'string'], [text, link]);
}
