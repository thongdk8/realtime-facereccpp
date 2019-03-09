#configure variables
set (APP_NAME "facetracker")

#configure directories
set (APP_PATH "${PROJECT_SOURCE_DIR}/facetracker")
set (APP_SRC_PATH  "${APP_PATH}/src" )
set (APP_INCLUDE_PATH "${APP_PATH}/include")

find_package(OpenCV REQUIRED)

set(Boost_USE_STATIC_LIBS   ON)
FIND_PACKAGE(Boost 1.58 COMPONENTS filesystem timer REQUIRED)

#set includes
include_directories (
    ${APP_INCLUDE_PATH}
    ${MTCNN_LIB_INCLUDE_PATH}
)

#set sources
file (GLOB APP_SOURCE_FILES "${APP_SRC_PATH}/*.cc")
file (GLOB APP_HEADER_FILES "${APP_INCLUDE_PATH}/*.h")

#set target executable
add_executable (${APP_NAME} ${APP_SOURCE_FILES} ${APP_HEADER_FILES})

if (WIN32)
set(LINK_OPTIONS shlwapi.lib)
endif()

if ("${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
set(LINK_OPTIONS -pthread -ldl -static-libgcc -static-libstdc++)
endif()

if (Darwin)
set(LINK_OPTIONS -pthread)
endif()

#add the library
target_link_libraries (
    ${APP_NAME}
    ${OpenCV_LIBRARIES}
    ${Boost_LIBRARIES}
    ${MTCNN_LIB_NAME}
    ${LINK_OPTIONS})

#test
enable_testing ()
