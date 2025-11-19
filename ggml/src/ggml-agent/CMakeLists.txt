message(STATUS "Building GGML Agent-to-Agent Protocol")

# Create the ggml-agent library
add_library(ggml-agent
    ggml-agent.cpp
    ../../include/ggml-agent.h
)

set_target_properties(ggml-agent PROPERTIES
    VERSION ${GGML_VERSION}
    SOVERSION ${GGML_VERSION_MAJOR}
)

# Link dependencies
target_link_libraries(ggml-agent PRIVATE ggml-base ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(ggml-agent PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_compile_features(ggml-agent PRIVATE cxx_std_17)

# Install
install(TARGETS ggml-agent LIBRARY)
install(FILES ../../include/ggml-agent.h DESTINATION include)
