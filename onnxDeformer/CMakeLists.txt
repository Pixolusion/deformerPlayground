project(inflateDeformerOnnx)
file(GLOB CPP_SOURCE_FILES "*.cpp")
file(GLOB CPP_HEADER_FILES "include/*.h*")

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
    ${CPP_SOURCE_FILES}
    ${CPP_HEADER_FILES}
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${MAYA_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    OpenMaya.lib
    OpenMayaAnim.lib
    Foundation.lib
    onnxruntime
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ${PLUGIN_EXTENSION}
)

# INSTALL
install(TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION ${MAYA_PLUGIN_DIR}
    LIBRARY DESTINATION ${MAYA_PLUGIN_DIR}
    RUNTIME DESTINATION ${MAYA_PLUGIN_DIR}
)
