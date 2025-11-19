# Agent Collaboration Examples

# Simple chat example
add_executable(simple-chat simple-chat.cpp)

target_link_libraries(simple-chat PRIVATE llama-agent)

target_compile_features(simple-chat PRIVATE cxx_std_17)

if(LLAMA_ALL_WARNINGS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(simple-chat PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    elseif(MSVC)
        target_compile_options(simple-chat PRIVATE /W4)
    endif()
endif()

# Installation
if(LLAMA_TOOLS_INSTALL)
    install(TARGETS simple-chat RUNTIME DESTINATION bin)
endif()
