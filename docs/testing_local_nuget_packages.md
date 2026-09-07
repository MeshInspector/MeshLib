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

### Copy the package to the local feed

The `./nuget` directory is called the "local feed", a place where nuget can install packages from.

To properly copy the package into it, run: `dotnet nuget push MeshLib_v1.2.3.4.nupkg --source "$(realpath nuget)"`.

The last parameter is the absolute path to this `nuget` directory, you can pass it manually instead of using `realpath`.

If done correctly, the package will be copied into `./nuget/MeshLib.1.2.3.4.nupkg` (notice it being automatically renamed). You can copy it yourself there too.

### Install package

Delete the existing MeshLib packages:

* On Windows, delete `C:\Users\user\.nuget\packages\meshlib`.

* On Linux: `rm -rf ~/.nuget/packages/meshlib/`

If this is a new project, add the dependency on MeshLib by adding the following part to the `.csproj` file:
```xml
<ItemGroup>
  <PackageReference Include="MeshLib" Version="1.2.3.4" />
</ItemGroup>
```

Run the project. NuGet will automatically install your package into `~/.nuget/packages/meshlib/1.2.3.4`.

Now the `./nuget` directory and the `NuGet.Config` file are no longer needed, and neither is the original package file that was passed to `dotnet nuget push ...`.
