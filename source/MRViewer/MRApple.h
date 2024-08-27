#pragma once

#ifdef __APPLE__

namespace MR
{

namespace Apple
{
	
// Register callback for a "documents open" event
// This event is sent in MacOS to the application when the user opens the files in Finder
// If the application is not active, it is started and then the event is sent: command line is not used for GUI apps
// So it must be installed before any events processing
void registerOpenDocumentsCallback();

}

}

#endif