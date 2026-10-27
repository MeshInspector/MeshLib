"""Inject stderr tracing into GLFW 3.4's wl_window.c (check build only). usage: glfw_debug_patch.py <wl_window.c>"""
import sys

p = sys.argv[1]
s = open(p).read()


def patch(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)


patch("""void _glfwUpdateBufferScaleFromOutputsWayland(_GLFWwindow* window)
{
""", """void _glfwUpdateBufferScaleFromOutputsWayland(_GLFWwindow* window)
{
    fprintf(stderr, "GLFWDBG updateBufferScale window=%p scaleFramebuffer=%d fractionalScale=%p outputs=%zu bufferScale=%d\\n",
            (void*)window, window->wl.scaleFramebuffer, (void*)window->wl.fractionalScale, window->wl.outputScaleCount, window->wl.bufferScale);
""")
patch("""static void surfaceHandleEnter(void* userData,
                               struct wl_surface* surface,
                               struct wl_output* output)
{
""", """static void surfaceHandleEnter(void* userData,
                               struct wl_surface* surface,
                               struct wl_output* output)
{
    fprintf(stderr, "GLFWDBG enter window=%p surface=%p output=%p tagged=%d\\n", userData, (void*)surface, (void*)output,
            wl_proxy_get_tag((struct wl_proxy*) output) == &_glfw.wl.tag);
""")
patch("""void fractionalScaleHandlePreferredScale(void* userData,
                                         struct wp_fractional_scale_v1* fractionalScale,
                                         uint32_t numerator)
{
""", """void fractionalScaleHandlePreferredScale(void* userData,
                                         struct wp_fractional_scale_v1* fractionalScale,
                                         uint32_t numerator)
{
    fprintf(stderr, "GLFWDBG preferred_scale window=%p numerator=%u\\n", userData, numerator);
""")
patch("""static GLFWbool createShellObjects(_GLFWwindow* window)
{
""", """static GLFWbool createShellObjects(_GLFWwindow* window)
{
    fprintf(stderr, "GLFWDBG createShellObjects window=%p surface=%p\\n", (void*)window, (void*)window->wl.surface);
""")
if "#include <stdio.h>" not in s:
    s = s.replace("#include <string.h>", "#include <string.h>\n#include <stdio.h>", 1)
open(p, "w").write(s)
print("GLFW debug patch applied")
