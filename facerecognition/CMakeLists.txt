#configure variables
set (REG_APP_NAME "facerecognition")

#configure directories
set (REG_APP_PATH "${PROJECT_SOURCE_DIR}/facerecognition")
set (REG_APP_SRC_PATH  "${REG_APP_PATH}/src" )
set (REG_APP_INCLUDE_PATH "${REG_APP_PATH}/include")
set (TRACKER_INCLUDE "${PROJECT_SOURCE_DIR}/facetracker/include")
set (DMLC_INCLUDE "/home/thongpb/works/tvm/3rdparty/dmlc-core/include")
set (DLPACK_INC "/home/thongpb/works/tvm/3rdparty/dlpack/include")

find_package(OpenCV REQUIRED)
# find_package(tvm REQUIRED)

set(Boost_USE_STATIC_LIBS   ON)
FIND_PACKAGE(Boost 1.58 COMPONENTS filesystem timer REQUIRED)

#set includes
include_directories (
    ${APP_INCLUDE_PATH}
    ${REG_APP_INCLUDE_PATH}
    ${MTCNN_LIB_INCLUDE_PATH}
    ${DMLC_INCLUDE}
    ${DLPACK_INC}
    ${TRACKER_INCLUDE}
)

#set sources
file (GLOB REG_APP_SOURCE_FILES "${REG_APP_SRC_PATH}/*.cc")
file (GLOB REG_APP_HEADER_FILES "${REG_APP_INCLUDE_PATH}/*.h")

#set library
add_library (tinyxml2 STATIC "${REG_APP_SRC_PATH}/tinyxml2.cc" "${REG_APP_INCLUDE_PATH}/tinyxml2.h")

#set target executable
add_executable (${REG_APP_NAME} ${REG_APP_SOURCE_FILES} ${REG_APP_HEADER_FILES})

if (WIN32)
set(LINK_OPTIONS shlwapi.lib)
endif()

if ("${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
set(LINK_OPTIONS -pthread -ldl -static-libgcc -static-libstdc++ -ltvm -ltvm_runtime -lnnvm_compiler -ltvm_topi)
endif()

if (Darwin)
set(LINK_OPTIONS -pthread)
endif()

#add the library
target_link_libraries (
    ${REG_APP_NAME}
    ${OpenCV_LIBRARIES}
    ${Boost_LIBRARIES}
    ${MTCNN_LIB_NAME}
    ${LINK_OPTIONS})

#test
enable_testing ()
