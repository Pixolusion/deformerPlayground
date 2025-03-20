find_package(Maya REQUIRED COMPONENTS maya)

if("${MAYA_VERSION}" STREQUAL "")
    message(FATAL_ERROR "Maya version is empty, find_package did not find this variable")
endif()

link_directories(
    ${MAYA_LIBRARY_DIR}
)
