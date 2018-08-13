project(graph)
cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)

#set(CMAKE_PREFIX_PATH ${LD_LIBRARY_PATH})

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RELEASE")
    #set(CMAKE_BUILD_TYPE "DEBUG")
    #set(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
    #set(CMAKE_BUILD_TYPE "MINSIZEREL")
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -DDEBUG -ftemplate-backtrace-limit=0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -ftemplate-backtrace-limit=0")
else()
    MESSAGE(FATAL_ERROR "CMakeLists.txt has not been tested/written for your compiler.")
endif()

#add_executable(subgraph graph.cpp nodeset.cpp utils.cpp subgraph.cpp exception.cpp)
#add_executable(graphfeat graph.cpp nodeset.cpp utils.cpp graphfeat.cpp exception.cpp)
#add_executable(centrality graph.cpp nodeset.cpp utils.cpp centrality.cpp exception.cpp)
#add_executable(_test_gfi _test_gfi.cpp graphfullindexed.cpp nodeset.cpp utils.cpp exception.cpp)
#add_executable(_test_mygraph _test_graph.cpp nodemap.cpp nodeset.cpp graph.cpp exception.cpp utils.cpp graph_pywrapper.cpp ioutils.cpp types.cpp)
add_library(mygraph SHARED nodemap.cpp nodeset.cpp graph.cpp exception.cpp utils.cpp graph_pywrapper.cpp ioutils.cpp types.cpp)
set_target_properties(mygraph PROPERTIES PREFIX "" )

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
#target_link_libraries(_test_mygraph python2.7)

find_package(Boost 1.54.0)
if(Boost_FOUND)
  include_directories("${Boost_INCLUDE_DIRS}" "/usr/include/python2.7")
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

#target_link_libraries(_test_mygraph ${Boost_LIBRARIES})
target_link_libraries(mygraph ${Boost_LIBRARIES})

#set_property(TARGET _test_mygraph mygraph PROPERTY CXX_STANDARD 11)
#set_property(TARGET _test_mygraph mygraph PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mygraph PROPERTY CXX_STANDARD 11)
set_property(TARGET mygraph PROPERTY CXX_STANDARD_REQUIRED ON)
