#list(APPEND srcs
#    ${CMAKE_CURRENT_SOURCE_DIR}/demo_ivfpq_indexing_gpu.cpp)

cmake_minimum_required(VERSION 3.15)

mark_as_advanced(
    BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
    gmock_build_tests gtest_build_samples gtest_build_tests
    gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
)

if (MSVC)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif(MSVC)

macro(gpufaiss_add_test TESTNAME)
    # create an exectuable in which the tests will be stored
    add_executable(${TESTNAME} ${ARGN})
	target_compile_definitions(${TESTNAME} PUBLIC GTEST_MAIN)
    # link the Google test infrastructure, mocking library, and a default main fuction to
    # the test executable.  Remove g_test_main if writing your own main function.
    target_link_libraries(${TESTNAME} gpufaiss faiss gtest gmock gtest_main ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
    # gtest_discover_tests replaces gtest_add_tests,
    # see https://cmake.org/cmake/help/v3.10/module/GoogleTest.html for more options to pass to it
    gtest_discover_tests(${TESTNAME}
        # set a working directory so your project root so that you can find test data via paths relative to the project root
        WORKING_DIRECTORY ${PROJECT_DIR}
        PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
    )
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()
file(GLOB_RECURSE  gpufaiss_test_sources ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
list(REMOVE_ITEM gpufaiss_test_sources ${CMAKE_CURRENT_SOURCE_DIR}/demo_ivfpq_indexing_gpu.cpp)

gpufaiss_add_test(gpufaiss_test ${gpufaiss_test_sources})
