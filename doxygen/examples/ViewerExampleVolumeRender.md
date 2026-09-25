# Viewer Example Volume Rendering {#ExampleViewerVolumeRender}

Example of using Viewer for Volume Rendering

<div class="tabbed">
 
- <b class="tab-title">Python</b>
> [!NOTE]
> Python API version 3 and later
 
\include ViewerVolumeRender.dox.py
\image html Volume_rendering.png
> [!NOTE]
> On macOS the Viewer runs on the main thread of the Python process, the only thread a GUI can run on there: `mv.launch()` prepares it, the calls that follow set up the scene, and the window appears in `mv.showViewer()`, which returns once the user closes it. Older releases raised `RuntimeError: MeshLib Viewer is not supported on macOS yet`, and before 3.1.3.566 terminated the Python process (SIGTRAP, exit code 133).

</div>
