#include "MRCustomImGuiConfig.h"


ImGuiContext*& MyImGuiTLS()
{
    static thread_local ImGuiContext* MyImGuiTLS_;
    return MyImGuiTLS_;
}
