cmake_minimum_required(VERSION 2.8)

project(algorithm)
#set(CMAKE_PREFIX_PATH ${LD_LIBRARY_PATH})

# use local find_packages
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RELEASE")
    #set(CMAKE_BUILD_TYPE "DEBUG")
    #set(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
    #set(CMAKE_BUILD_TYPE "MINSIZEREL")
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -DDEBUG -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
else()
  MESSAGE(FATAL_ERROR "CMakeLists.txt has not been tested/written for your compiler.")
endif()

add_library(dynamic_triad SHARED dynamic_triad_cimpl.cpp)
set_target_properties(dynamic_triad PROPERTIES PREFIX "" OUTPUT_NAME "dynamic_triad_cimpl")

#add_library(dynamic_triad_extfeat SHARED dynamic_triad_extfeat_cimpl.cpp)
#set_target_properties(dynamic_triad_extfeat PROPERTIES PREFIX "" OUTPUT_NAME "dynamic_triad_extfeat_cimpl")
#add_library(resolve_pyapi SHARED resolve.cpp resolve_pywrapper.cpp)
#add_library(logfile_capi SHARED logfile_capi.cpp logfile.cpp)
#add_library(unitfeat_pyapi SHARED unit_feat_pywrapper.cpp)
#add_executable(test_logfile logfile.cpp test_logfile.cpp)
#set_target_properties(resolve_pyapi PROPERTIES PREFIX "" OUTPUT_NAME "cresolve")
#set_target_properties(unitfeat_pyapi PROPERTIES PREFIX "" OUTPUT_NAME "cunitfeat")

#add_library(gounitfeat STATIC IMPORTED)
#set_target_properties(gounitfeat PROPERTIES
#    IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/libgounitfeat.a)

#include_directories(${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR})
#link_directories(${CMAKE_BINARY_DIR})
#target_link_libraries(unitfeat_pyapi gounitfeat logfile_capi)

#if(CMAKE_STATIC_GLIB_PATH)
#    find_package(Glibc 3.14 REQUIRED PATH ${CMAKE_STATIC_GLIBC_PATH}
#    NO_SYSTEM_ENVIRONMENT_PATH)
#    target_link_libraries(cppimpl ${GLIBC_LIBRARIES})
#    target_link_libraries(mygraph ${GLIBC_LIBRARIES})
#    set_target_properties(cppimpl mygraph PROPERTIES LINK_SEARCH_START_STATIC 1)
#    set_target_properties(cppimpl mygraph PROPERTIES LINK_SEARCH_END_STATIC 1)
#endif()

# we MUST find python before boost_python,
# because boost python somehow sets PYTHON_EXECUTABLE in a wrong manner
# and find_package(python) will not reset this variable once it exists
# as a result, find_package(python) is broken due to wrong path

# assume the project main directory is ..
include_directories(${CMAKE_SOURCE_DIR}/..)

# respect user environment if the following variables are set
if((NOT "$ENV{PYTHON_LIBRARY}" STREQUAL "") AND (NOT
    "$ENV{PYTHON_INCLUDE_DIR}" STREQUAL ""))
    message("Using custom python path $ENV{PYTHON_LIBRARY} and $ENV{PYTHON_INCLUDE_DIR}")
    set(PYTHON_LIBRARY $ENV{PYTHON_LIBRARY})
    set(PYTHON_INCLUDE_DIR $ENV{PYTHON_INCLUDE_DIR})
endif()

set(Python_ADDITIONAL_VERSIONS 2.7)
find_package(PythonLibs 2.7 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
link_directories(${PYTHON_LIBRARIES})
target_link_libraries(dynamic_triad python2.7)
#target_link_libraries(dynamic_triad_extfeat python2.7)

# pythoninterp required by findnumpy
find_package(PythonInterp 2.7 REQUIRED)
find_package(NumPy 1.10 REQUIRED)
if(NOT ${PYTHON_NUMPY_FOUND})
    message(FATAL_ERROR "Cannot find numpy headers")
endif()
include_directories(${PYTHON_NUMPY_INCLUDE_DIR})

if("$ENV{EIGEN3_INCLUDE_DIR}" STREQUAL "")
    message(WARNING "EIGEN3_INCLUDE_DIR not set, trying to find eigen headers in standard directories")
else()
    message("Using eigen path $ENV{EIGEN3_INCLUDE_DIR}")
    include_directories("$ENV{EIGEN3_INCLUDE_DIR}")
endif()

find_package(Boost 1.54.0)
if(Boost_FOUND)
  include_directories("${Boost_INCLUDE_DIRS}")
  set(Boost_USE_STATIC_LIBS OFF)
  set(Boost_USE_MULTITHREADED ON)
  set(Boost_USE_STATIC_RUNTIME OFF)
  find_package(Boost 1.54.0 COMPONENTS python)
  if(NOT Boost_FOUND)
      if(NOT "$ENV{BOOST_PYTHON_LIBNAME}" STREQUAL "")
          message("boost_python not detected by cmake, trying custom lib name
          $ENV{BOOST_PYTHON_LIBNAME}")
      else()
          message(FATAL_ERROR "boost_python not detected by cmake, try setting
          environment variable $BOOST_PYTHON_LIBNAME for custom library name")
      endif()
      set (Boost_LIBRARIES "${Boost_LIBRARIES}-l boost_python-2.7")
  endif()
ELSEIF(NOT Boost_FOUND)
  MESSAGE(FATAL_ERROR "Unable to find correct Boost version. Did you set BOOST_ROOT?")
ENDIF()
target_link_libraries(dynamic_triad ${Boost_LIBRARIES})
#target_link_libraries(dynamic_triad_extfeat ${Boost_LIBRARIES})

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
  message("OPENMP FOUND")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set_property(TARGET dynamic_triad PROPERTY CXX_STANDARD 11)
set_property(TARGET dynamic_triad PROPERTY CXX_STANDARD_REQUIRED ON)

install(TARGETS dynamic_triad DESTINATION ${CMAKE_SOURCE_DIR} LIBRARY)
