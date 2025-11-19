# Agent Collaboration Framework

set(TARGET llama-agent)

add_library(${TARGET} STATIC
    message.cpp
    conversation.cpp
    agent.cpp
    registry.cpp
    failure.cpp
)

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor
)

target_link_libraries(${TARGET} PRIVATE
    ${CMAKE_THREAD_LIBS_INIT}
)

# C++17 required for std::optional and other features
target_compile_features(${TARGET} PUBLIC cxx_std_17)

# Enable warnings
if(LLAMA_ALL_WARNINGS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${TARGET} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wno-unused-parameter
        )
    elseif(MSVC)
        target_compile_options(${TARGET} PRIVATE /W4)
    endif()
endif()

# Installation
install(TARGETS ${TARGET}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES
    message.h
    conversation.h
    agent.h
    registry.h
    failure.h
    DESTINATION include/llama/agent
)
