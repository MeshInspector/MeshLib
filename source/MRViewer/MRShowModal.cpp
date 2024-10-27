#include "MRShowModal.h"
#include "ImGuiMenu.h"
#include <MRPch/MRSpdlog.h>

namespace MR
{

void showModal( const std::string& msg, NotificationType type )
{
    if ( auto menu = ImGuiMenu::instance() )
        menu->showModalMessage( msg, type );
    else
    {
        if ( type == NotificationType::Error )
            spdlog::error( "Show Error: {}", msg );
        else if ( type == NotificationType::Warning )
            spdlog::warn( "Show Warning: {}", msg );
        else //if ( type == MessageType::Info )
            spdlog::info( "Show Info: {}", msg );
    }
}

} //namespace MR
