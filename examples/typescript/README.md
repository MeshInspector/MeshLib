# MeshLib TypeScript example

A single-file TypeScript program that uses the `@meshinspector/meshlib` WebAssembly bindings:
builds meshes from typed arrays, runs a boolean and a nearest-point query, and reads geometry
back — all fully typed.

## Build and run

Assumes `@meshinspector/meshlib` is installed. From this directory:

```sh
npm install     # typescript (and the bindings, if not already present)
npm start       # build + type-check (tsc via prestart), then run (node)
```

For the multi-threaded build, install `@meshinspector/meshlib-mt` and change the import in
`main.ts`; the API is identical.
