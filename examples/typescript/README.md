# MeshLib TypeScript example

`main.ts` is a TypeScript program that uses the `@meshinspector/meshlib` WebAssembly bindings:
builds meshes from typed arrays, runs a boolean and a nearest-point query, and reads geometry
back — all fully typed.

`using-snippet.ts` is the snippet from the *TypeScript* section of the
[JavaScript Setup Guide](https://meshlib.io/documentation/MeshLibJsSetupGuide.html), kept here so
`tsc` type-checks it and it cannot rot. It is compiled but not run: `using` needs Node.js 24+.

## Build and run

Assumes `@meshinspector/meshlib` is installed. From this directory:

```sh
npm install     # typescript (and the bindings, if not already present)
npm start       # build + type-check (tsc via prestart), then run (node)
```

For the multi-threaded build, install `@meshinspector/meshlib-mt` and change the imports in
`main.ts` and `using-snippet.ts`; the API is identical.
