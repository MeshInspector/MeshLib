#pragma once

#ifdef __APPLE__

namespace MR
{

namespace Apple
{
	
// Register callback for "documents open" event, when the user opens the files in Finder
// MacOS does not use command line for GUI apps, it starts the application (unless already open) and then sends this event
// So it must be installed before any events processing
void registerOpenDocumentsCallback();

}

}

#endif
