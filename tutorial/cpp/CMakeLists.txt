file(GLOB cpu_srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB gpu_srcs ${CMAKE_CURRENT_SOURCE_DIR}/*GPU*.cpp)

# Build each source file independently
include_directories(../../../)	# faiss root directory

if(BUILD_WITH_GPU)
	foreach(source ${gpu_srcs})
		list(REMOVE_ITEM cpu_srcs ${source})
	endforeach(source)
	#list(REMOVE_ITEM srcs ${CMAKE_CURRENT_SOURCE_DIR}/4-GPU.cpp)
	#list(REMOVE_ITEM srcs ${CMAKE_CURRENT_SOURCE_DIR}/5-Multiple-GPUs.cpp)
endif(BUILD_WITH_GPU)

foreach(source ${cpu_srcs})
    get_filename_component(name ${source} NAME_WE)

    # target
    add_executable(${name} ${source})
	target_link_libraries(${name} ${faiss_lib})

    # Install
    install(TARGETS ${name} DESTINATION bin)
endforeach(source)

if(BUILD_WITH_GPU)
foreach(source ${gpu_srcs})
    get_filename_component(name ${source} NAME_WE)

    # target
    add_executable(${name} ${source})
	target_link_libraries(${name} ${faiss_lib} ${faiss_lib_gpu} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

    # Install
    install(TARGETS ${name} DESTINATION bin)
endforeach(source)
endif(BUILD_WITH_GPU)
