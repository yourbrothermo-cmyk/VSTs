add_executable(voxforge_tests test_main.cpp)
target_link_libraries(voxforge_tests PRIVATE voxforge_core)
add_test(NAME voxforge_tests COMMAND voxforge_tests)

add_executable(voxforge_cli cli.cpp)
target_link_libraries(voxforge_cli PRIVATE voxforge_core)
