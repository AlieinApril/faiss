file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Build each source file independently
include_directories(../../../)	# faiss root directory

if(NOT BUILD_WITH_GPU)
	list(REMOVE_ITEM srcs ${CMAKE_CURRENT_SOURCE_DIR}/4-GPU.cpp)
	list(REMOVE_ITEM srcs ${CMAKE_CURRENT_SOURCE_DIR}/5-Multiple-GPUs.cpp)
endif(NOT BUILD_WITH_GPU)

foreach(source ${srcs})
    get_filename_component(name ${source} NAME_WE)

    # target
    add_executable(${name} ${source})
    target_link_libraries(${name} ${faiss_lib})

    # Install
    install(TARGETS ${name} DESTINATION bin)
endforeach(source)
