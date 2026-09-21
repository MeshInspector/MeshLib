# Viewer Example {#ExampleViewer}

Example of using Viewer

<div class="tabbed">
 
- <b class="tab-title">Python</b>
> [!NOTE]
> Python API version 3 and later
 
\include Viewer.dox.py
Viewer with scene tree
\image html Viewer_with_scene_tree.png
Viewer without scene tree
\image html Viewer_without_scene_tree.png
> [!NOTE]
> On macOS the Viewer runs on the main thread of the Python process, the only thread a GUI can run on there: `mv.launch()` creates the window, it stays live while Python waits for terminal input - at the interactive prompt or in `input()` - and `mv.showViewer()` hands it to the user until they close it. Older releases raised `RuntimeError: MeshLib Viewer is not supported on macOS yet`, and before 3.1.3.566 terminated the Python process (SIGTRAP, exit code 133).

</div>
