cmake_minimum_required(VERSION 3.14)

project(deformerCompare)

if (NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    if (WIN32)
        message(FATAL_ERROR "Compiler does not support 64-bit builds. "
                "If you are using Visual Studio, make sure you are in the "
                "x64 command prompt and have specified the Win64 cmake "
                "generator (e.g., -G \"Visual Studio 15 2017 Win64\")")
    else()
        message(FATAL_ERROR "Compiler does not support 64-bit builds.")
    endif()
endif()

list(APPEND CMAKE_MODULE_PATH
    ${PROJECT_SOURCE_DIR}/cmake/defaults
    ${PROJECT_SOURCE_DIR}/cmake/modules
    ${PROJECT_SOURCE_DIR}/cmake/macros
    $ENV{MAYA_DEVKIT}/cmake/modules
)

include(Options)
include(ProjectDefaults)
include(Packages)

# CXXDefaults will set a variety of variables for the project.
# Consume them here. This is an effort to keep the most common
# build files readable.
include(CXXDefaults)

set(INSTALL_DIR $ENV{USERPROFILE}/Documents/maya/${MAYA_VERSION})
string(REPLACE "\\" "/" INSTALL_DIR $ENV{USERPROFILE}/Documents/maya/${MAYA_VERSION})
set(PLUGIN_EXTENSION ".mll")
set(MAYA_SCRIPT_DIR ${INSTALL_DIR}/scripts)
set(MAYA_PLUGIN_DIR ${INSTALL_DIR}/plug-ins)
set(MAYA_ICONS_DIR ${INSTALL_DIR}/prefs/icons)

add_subdirectory(cppDeformer)
add_subdirectory(onnxDeformer)
