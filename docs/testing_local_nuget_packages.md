## How to install `.nuget` packages locally

This was tested on Linux, but should work the same way on Windows too.

### Create NuGet config

Create a file called `NuGet.Config` in the same directory as the `.csproj` project, with following contents:

```xml
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <add key="local-packages" value="nuget" />
  </packageSources>
</configuration>
```

Here `"local-packages"` can be any string, and `"nuget"` is the directory name where you'll put the package, relative to the current directory (where this config file is placed).

### Create the local feed

The `./nuget` directory is called the "local feed", a place where nuget can install packages from.

In theory, the contents must be created using the `nuget ...` command, but installing it on Linux is tricky, so we'll do it manually.

We want the following hierarchy: (replace `1.2.3.4` with your version)

```
nuget
└── meshlib
    └── 1.2.3.4
        ├── meshlib.1.2.3.4.nupkg
        ├── meshlib.1.2.3.4.nupkg.sha512
        └── MeshLib.nuspec
```

Create the directory `nuget/meshlib/1.2.3.4`. Note that the package name must be in lowercase.

Copy your `MeshLib_v1.2.3.4.nupkg` into `nuget/meshlib/1.2.3.4/meshlib.1.2.3.4.nupkg`. Note that the package name must be changed to lowercase, and `_v` replaced with `.`.

Create the `.sha512` file using:
```sh
sha512sum nuget/meshlib/1.2.3.4/meshlib.1.2.3.4.nupkg >nuget/meshlib/1.2.3.4/meshlib.1.2.3.4.nupkg.sha512
```

`MeshLib.nuspec` must be unzipped from `.nupkg`, which is just a ZIP archive. Do that using:
```sh
unzip nuget/meshlib/1.2.3.4/meshlib.1.2.3.4.nupkg -d nuget/meshlib/1.2.3.4/ MeshLib.nuspec
```
### Install package

Delete the existing MeshLib packages using:
```sh
rm -rf ~/.nuget/packages/meshlib/
```

Run the project. NuGet will automatically install your package into `~/.nuget/packages/meshlib/1.2.3.4`.

Now the `./nuget` directory and the `NuGet.Config` file are no longer needed.
