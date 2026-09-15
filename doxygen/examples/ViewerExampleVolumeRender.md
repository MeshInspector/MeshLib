# Viewer Example Volume Rendering {#ExampleViewerVolumeRender}

Example of using Viewer for Volume Rendering

<div class="tabbed">
 
- <b class="tab-title">Python</b>
> [!NOTE]
> Python API version 3 and later
 
\include ViewerVolumeRender.dox.py
\image html Volume_rendering.png
> [!NOTE]
> The example drives the Viewer from a function passed as `mv.launch(script=main)`: the window runs on the main thread, the only thread macOS allows a GUI on, and `main` on a worker thread. This form works on every platform; the plain `mv.launch()` followed by viewer calls works on Windows and Linux only, and on macOS raises a `RuntimeError` naming the `script` form (releases before 3.1.3.566 terminated the Python process instead, SIGTRAP, exit code 133).

</div>
