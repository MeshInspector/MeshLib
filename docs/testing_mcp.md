# How to test MCP?

Use "MCP Inspector". Run it with `npx @modelcontextprotocol/inspector`, where `npx` is installed as a part of Node.JS.

Set:
* Transport Type = SSE
* URL = http://localhost:8080/sse
* Connection Type = Via Proxy  (Doesn't work for me without proxy now when we're using the Fastmcpp library, but did work with another library; not sure why.)

Press `Connect`.

Press `List Tools` (if grayed out, do `Clear` first).

Click on your tool.

On the right panel, set parameters. For some parameter types, it helps to press `Switch to JSON` on the right, then type them as JSON.

Press `Run Tool`. If you get weird errors, try pressing it again. In some cases, the first press passes stale/empty parameters.

Then check for validation errors, below this button.

If it complains that your output doesn't match the schema you specified, paste both the output and the schema (using the `Copy` button in the top-right corner of the code blocks; that copies JSON properly, unlike Ctrl+C in this case)
  into a schema validator, e.g. https://www.jsonschemavalidator.net/

**NOTE:** It doesn't seem to validate the input schema (only output schema). Check it by eye.
