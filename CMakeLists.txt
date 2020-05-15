cmake_minimum_required(VERSION 3.1.0)

project(OpenNMTTokenizer)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

set(INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}
  )

set(PUBLIC_HEADERS
  include/onmt/Alphabet.h
  include/onmt/AnnotatedToken.h
  include/onmt/BPE.h
  include/onmt/BPELearner.h
  include/onmt/CaseModifier.h
  include/onmt/ITokenizer.h
  include/onmt/SpaceTokenizer.h
  include/onmt/SubwordEncoder.h
  include/onmt/SubwordLearner.h
  include/onmt/Tokenizer.h
  )

set(SOURCES
  src/Alphabet.cc
  src/AnnotatedToken.cc
  src/BPE.cc
  src/BPELearner.cc
  src/CaseModifier.cc
  src/ITokenizer.cc
  src/SpaceTokenizer.cc
  src/SubwordEncoder.cc
  src/SubwordLearner.cc
  src/Tokenizer.cc
  src/unicode/Unicode.cc
  )

list(APPEND LINK_LIBRARIES "")
list(APPEND SOURCES src/unicode/Data.cc)

add_library(${PROJECT_NAME} STATIC ${SOURCES})
include(GNUInstallDirs)
include(GenerateExportHeader)
string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWER)
generate_export_header(${PROJECT_NAME} EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/onmt/${PROJECT_NAME_LOWER}_export.h)
target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIRECTORIES})
target_link_libraries(${PROJECT_NAME} ${LINK_LIBRARIES})
