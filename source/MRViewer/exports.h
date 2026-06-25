#pragma once

// see explanation in MRMesh/MRMeshFwd.h
#ifdef _WIN32
#   ifdef MRViewer_EXPORTS
#       define MRVIEWER_API __declspec(dllexport)
#   else
#       define MRVIEWER_API __declspec(dllimport)
#   endif
#   define MRVIEWER_CLASS
#else
#   define MRVIEWER_API   __attribute__((visibility("default")))
#   ifdef __clang__
#       define MRVIEWER_CLASS __attribute__((type_visibility("default")))
#   else
#       define MRVIEWER_CLASS __attribute__((visibility("default")))
#   endif
#endif

// Note! This macro should be used in source files only!
# define MRVIEWER_PLUGIN_REGISTRATION(pluginClassName) \
struct Instance##pluginClassName \
{ \
    Instance##pluginClassName() \
    { \
        MR::Viewer::instance()->plugins.push_back( &plugin ); \
    } \
    pluginClassName plugin; \
}; \
static Instance##pluginClassName instance##pluginClassName;

// Note! This macro should be used in source files only!
# define MRVIEWER_PLUGIN_APPLY_FUNCTION(pluginClassName, func) \
struct Function##func##pluginClassName \
{ \
    Function##func##pluginClassName() \
    { \
        auto pluginInstance = MR::Viewer::instance()->getPluginInstance<pluginClassName>(); \
        func(pluginInstance); \
    } \
}; \
static Function##func##pluginClassName function##func##pluginClassName;
