var customSetCursor = function (cursor) {
    switch (cursor) {
        //ImGuiMouseCursor_None
        case -1:
            {
                Module["canvas"].style.cursor = "none";
                break
            }
        //ImGuiMouseCursor_Arrow
        case 0:
            {
                Module["canvas"].style.cursor = "default";
                break
            }
        //ImGuiMouseCursor_TextInput
        case 1:
            {
                Module["canvas"].style.cursor = "text";
                break
            }
        //ImGuiMouseCursor_ResizeAll
        case 2:
            {
                Module["canvas"].style.cursor = "all-scroll";
                break
            }
        //ImGuiMouseCursor_ResizeNS
        case 3:
            {
                Module["canvas"].style.cursor = "ns-resize";
                break
            }
        //ImGuiMouseCursor_ResizeEW
        case 4:
            {
                Module["canvas"].style.cursor = "ew-resize";
                break
            }
        //ImGuiMouseCursor_ResizeNESW
        case 5:
            {
                Module["canvas"].style.cursor = "nesw-resize";
                break
            }
        //ImGuiMouseCursor_ResizeNWSE
        case 6:
            {
                Module["canvas"].style.cursor = "nwse-resize";
                break
            }
        //ImGuiMouseCursor_Hand
        case 7:
            {
                Module["canvas"].style.cursor = "pointer";
                break
            }
        //ImGuiMouseCursor_NotAllowed
        case 8:
            {
                Module["canvas"].style.cursor = "not-allowed";
                break
            }
        default:
            {
                Module["canvas"].style.cursor = "default";
                break
            }
    }
};
