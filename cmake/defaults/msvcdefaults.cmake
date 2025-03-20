
set(CMAKE_CONFIGURATION_TYPES "Release;Debug" CACHE STRING "" FORCE)

# reduce flags to get control over what cmake produces and to allow commenting/descriptions
set(CMAKE_CXX_FLAGS "/DWIN32 /D_WINDOWS")  # recorded default "/DWIN32 /D_WINDOWS /W3 /GR"
set(CMAKE_CXX_FLAGS_RELEASE "/DNDEBUG")  # recorded default "/MD /O2 /Ob2 /DNDEBUG", reducing flags to allow flag commenting below
set(CMAKE_CXX_FLAGS_DEBUG "/DDEBUG /MDd /Zi /Ob0 /Od /RTC1")  # recorded default "/MDd /Zi /Ob0 /Od /RTC1"
add_definitions(-DWINDOWS)
add_definitions(-D_AMD64_)  # This is normally defined by windows.h, defining here allows us to include smaller headers.
add_definitions(-DWIN32_LEAN_AND_MEAN)
add_definitions(-D_HAS_EXCEPTIONS=0)  # Turn off "disabled c++ exceptions" warning for the c++ standard library
add_compile_options("$<$<CONFIG:RELEASE>:/O2>")  # Enable Maximum optimisation in favor of speed of code (no O3, highest for msvc)
add_compile_options("$<$<CONFIG:RELEASE>:/Ob2>")  # Enable inline function expansion - any suitable
if (CPP_WARNINGS)
    add_compile_options("/W3")  # Warnings upto level 3
else()
    add_compile_options("/W0")
endif()

# As we are deploying the libraries at 64-bit the default is SSE2 and this becomes an unrecognised option for msvc x64
# add_compile_options("/arch:SSE2")  # Enable Streaming SIMD extensions 2
add_compile_options("/Oi")  # Enable intrinsics
add_compile_options("/Ot")  # Enable favor speed code
add_compile_options("/Zc:externConstexpr")  # confrom to the C++ standard and allow external linkage for constexpr
add_definitions(-DNOMINMAX)
