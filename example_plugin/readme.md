# TODO: WRITE README
Please do not duplicate https://github.com/MeshInspector/MeshLib?tab=readme-ov-file#distribution, also don't forget about:
 - /Zc:preprocessor
 - /bigobj /utf-8
 - c++ 20+
 - And this:
```xml
   <ItemGroup>
    <CopyFileToFolders Include="MyPlugin.items.json">
      <FileType>Document</FileType>
    </CopyFileToFolders>
    <CopyFileToFolders Include="MyPlugin.ui.json">
      <FileType>Document</FileType>
    </CopyFileToFolders>
    <Content Include="resource\**">
      <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
    </Content>
  </ItemGroup>
```