cmake_minimum_required(VERSION 3.15)
project(ekaterinburg)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DVK_PROTOTYPES")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_PROTOTYPES")

# GLFW
set(GLFW_DIR lib/glfw)
option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
option(GLFW_INSTALL "Generate installation target" OFF)
option(GLFW_DOCUMENT_INTERNALS "Include internals in documentation" OFF)
add_subdirectory(${GLFW_DIR} binary_dir EXCLUDE_FROM_ALL)
include_directories(${GLFW_DIR}/include)

# Dear ImGui
set(IMGUI_DIR lib/imgui)
include_directories(${IMGUI_DIR} ${IMGUI_DIR}/backends ..)

# Libraries
find_package(Vulkan REQUIRED)
set(LIBRARIES "glfw;Vulkan::Vulkan")

# Use vulkan headers from glfw:
include_directories(${GLFW_DIR}/deps)

file(GLOB sources *.cpp)

function(build_shader TARGETOBJ SHADERNAME)
    set(FILENAME "${CMAKE_SOURCE_DIR}/${SHADERNAME}")
    set(OUTNAME "${CMAKE_BINARY_DIR}/${SHADERNAME}.spv")
    add_custom_command(OUTPUT ${OUTNAME}
            COMMAND glslc ${FILENAME} -o ${OUTNAME}
            DEPENDS ${FILENAME})
    add_custom_target(MAKE_${SHADERNAME} ALL DEPENDS ${OUTNAME})
    add_dependencies(${TARGETOBJ} MAKE_${SHADERNAME})
endfunction()

add_executable(vulkan_tutorial vulkan_tutorial.cpp)
target_link_libraries(vulkan_tutorial ${LIBRARIES})

build_shader(vulkan_tutorial shader.frag)
build_shader(vulkan_tutorial shader.vert)
