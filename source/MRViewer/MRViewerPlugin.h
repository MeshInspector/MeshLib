#pragma once
// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2014 Daniele Panozzo <daniele.panozzo@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

// TODO:
// * create plugins/skeleton.h
// * pass time in draw function
// * remove Preview3D from comments
// * clean comments

#include "MRViewerInstance.h"
#include <filesystem>
#include <vector>

namespace MR
{

// Abstract class for plugins
// All plugins MUST have this class as their parent and may implement any/all
// the callbacks marked `virtual` here.
//
// /////For an example of a basic plugins see plugins/skeleton.h
//
// Return value of callbacks: returning true to any of the callbacks tells
// Viewer that the event has been handled and that it should not be passed to
// other plugins or to other internal functions of Viewer

class ViewerPlugin
{
public:
    ViewerPlugin()
    {
        plugin_name = "dummy";
    }

    virtual ~ViewerPlugin()
    {
    }

    // This function is called when the viewer is initialized (no mesh will be loaded at this stage)
    virtual void init( Viewer *_viewer )
    {
        viewer = _viewer;
    }

    // This function is called before shutdown
    virtual void shutdown()
    {
    }

    std::string plugin_name;
protected:
    // Pointer to the main Viewer class
    Viewer *viewer = &getViewerInstance();
};

}
