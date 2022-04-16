cmake_minimum_required(VERSION 3.1.0)

project(OpenNMTTokenizer)

option(BUILD_TESTS "Compile unit tests" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

set(CMAKE_CXX_STANDARD 11)
if(CMAKE_VERSION VERSION_LESS "3.7.0")
  set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

option(SPM_ENABLE_SHARED "" OFF)
option(SPM_ENABLE_TCMALLOC "" OFF)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/sentencepiece EXCLUDE_FROM_ALL)

if(MSVC)
  set(SENTENCEPIECE_CXX_FLAGS "/w")
else()
  set(SENTENCEPIECE_CXX_FLAGS "-Wno-all -Wno-extra -Wno-error -Wno-stringop-overflow")
endif()

set_target_properties(
  sentencepiece-static
  sentencepiece_train-static
  PROPERTIES COMPILE_FLAGS ${SENTENCEPIECE_CXX_FLAGS}
  )

if(MSVC)
  if(NOT BUILD_SHARED_LIBS)
    if(CMAKE_VERSION VERSION_LESS "3.15.0")
      message(FATAL_ERROR "Use CMake 3.15 or later when setting BUILD_SHARED_LIBS to OFF")
    endif()
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /wd4251 /d2FH4-")
  set(ICU_COMPONENTS uc dt)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
  set(ICU_COMPONENTS uc data)
endif()

if(NOT DEFINED ICU_INCLUDE_DIRS OR NOT DEFINED ICU_LIBRARIES)
  find_package(ICU REQUIRED COMPONENTS ${ICU_COMPONENTS})
endif()

set(INCLUDE_DIRECTORIES
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}

  PRIVATE
  ${ICU_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sentencepiece/src
  )

set(PUBLIC_HEADERS
  include/onmt/Token.h
  include/onmt/BPE.h
  include/onmt/BPELearner.h
  include/onmt/ITokenizer.h
  include/onmt/SPMLearner.h
  include/onmt/SentencePiece.h
  include/onmt/SentencePieceLearner.h
  include/onmt/SubwordEncoder.h
  include/onmt/SubwordLearner.h
  include/onmt/Tokenizer.h
  include/onmt/Vocab.h
  )

set(SOURCES
  src/BPE.cc
  src/BPELearner.cc
  src/Casing.cc
  src/ITokenizer.cc
  src/SentencePiece.cc
  src/SentencePieceLearner.cc
  src/SubwordEncoder.cc
  src/SubwordLearner.cc
  src/Token.cc
  src/Tokenizer.cc
  src/Utils.cc
  src/Vocab.cc
  src/unicode/Unicode.cc
  )

list(APPEND LINK_LIBRARIES
  ${ICU_LIBRARIES}
  sentencepiece-static
  sentencepiece_train-static
  )
list(APPEND SOURCES src/unicode/Data.cc)

if(COMMAND create_library)
  create_library(${PROJECT_NAME} SHARED ${SOURCES})
else()
  add_library(${PROJECT_NAME} SHARED ${SOURCES})
endif()

find_library(SP_LIBRARY NAMES sentencepiece)
find_path(SP_INCLUDE_DIR NAMES sentencepiece_processor.h)
if(NOT SP_LIBRARY OR NOT SP_INCLUDE_DIR)
    message(WARNING "sentencepiece not found; will not be supported")
else()
    message(STATUS "Found sentencepiece: ${SP_LIBRARY}")
    add_definitions(-DWITH_SP)
    list(APPEND PUBLIC_HEADERS include/onmt/SentencePiece.h)
    list(APPEND SOURCES src/SentencePiece.cc)
    list(APPEND INCLUDE_DIRECTORIES ${SP_INCLUDE_DIR})
    list(APPEND LINK_LIBRARIES ${SP_LIBRARY})
    file(STRINGS ${SP_INCLUDE_DIR}/sentencepiece_processor.h HAS_SAMPLE_ENCODE REGEX "SampleEncode")
    if(HAS_SAMPLE_ENCODE)
        add_definitions(-DSP_HAS_SAMPLE_ENCODE)
    endif()
    file(STRINGS ${SP_INCLUDE_DIR}/sentencepiece_processor.h HAS_VOCAB_RESTRICTION REGEX "SetVocabulary")
    if(HAS_VOCAB_RESTRICTION)
        add_definitions(-DSP_HAS_VOCAB_RESTRICTION)
    endif()
endif()

add_library(${PROJECT_NAME} STATIC ${SOURCES})
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE True)

include(GNUInstallDirs)
include(GenerateExportHeader)
string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWER)
generate_export_header(${PROJECT_NAME} EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/onmt/${PROJECT_NAME_LOWER}_export.h)
target_include_directories(${PROJECT_NAME} ${INCLUDE_DIRECTORIES})
target_link_libraries(${PROJECT_NAME} ${LINK_LIBRARIES})

if (NOT LIB_ONLY)
  add_subdirectory(cli)
endif()

if (BUILD_TESTS)
  add_subdirectory(test)
endif()

install(
  TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
install(FILES
  ${PUBLIC_HEADERS} "${PROJECT_BINARY_DIR}/onmt/${PROJECT_NAME_LOWER}_export.h"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onmt"
  )
install(FILES
  include/onmt/unicode/Unicode.h
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onmt/unicode"
  )
