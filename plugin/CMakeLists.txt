include(FetchContent)
set(JUCE_VERSION 8.0.4 CACHE STRING "JUCE tag to fetch")
if(DEFINED JUCE_DIR AND EXISTS "${JUCE_DIR}/CMakeLists.txt")
  add_subdirectory(${JUCE_DIR} ${CMAKE_BINARY_DIR}/juce)
else()
  FetchContent_Declare(juce GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
                            GIT_TAG ${JUCE_VERSION} GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(juce)
endif()

set(VOXFORGE_FORMATS VST3 Standalone)
if(APPLE)
  list(PREPEND VOXFORGE_FORMATS AU) # Logic Pro loads Audio Units only
endif()

juce_add_plugin(VoxForge
  COMPANY_NAME "VoxForge"
  BUNDLE_ID "com.voxforge.voxforge"
  PLUGIN_MANUFACTURER_CODE Vxfg
  PLUGIN_CODE Vxf1
  FORMATS ${VOXFORGE_FORMATS}
  PRODUCT_NAME "VoxForge"
  IS_SYNTH FALSE
  NEEDS_MIDI_INPUT FALSE
  NEEDS_MIDI_OUTPUT FALSE
  IS_MIDI_EFFECT FALSE
  EDITOR_WANTS_KEYBOARD_FOCUS FALSE
  AU_MAIN_TYPE kAudioUnitType_Effect
  VST3_CATEGORIES Fx Pitch Shift
  COPY_PLUGIN_AFTER_BUILD FALSE)

juce_generate_juce_header(VoxForge)

target_sources(VoxForge PRIVATE
  Source/PluginProcessor.cpp
  Source/PluginEditor.cpp
  Source/StudioSession.cpp)

target_compile_definitions(VoxForge PUBLIC
  JUCE_WEB_BROWSER=0
  JUCE_USE_CURL=0
  JUCE_VST3_CAN_REPLACE_VST2=0
  JUCE_DISPLAY_SPLASH_SCREEN=0)

target_link_libraries(VoxForge
  PRIVATE voxforge_core juce::juce_audio_utils juce::juce_audio_formats
  PUBLIC juce::juce_recommended_config_flags juce::juce_recommended_lto_flags juce::juce_recommended_warning_flags)

# Ship ONNX Runtime inside each plugin bundle so the plugin is self-contained.
if(VOXFORGE_WITH_ONNX AND APPLE)
  # Copy the versioned dylib (its install name is @rpath/libonnxruntime.X.Y.Z.dylib).
  file(REAL_PATH "${ORT_LIB}" ORT_LIB_REAL)
  foreach(fmt ${VOXFORGE_FORMATS})
    set(tgt VoxForge_${fmt})
    if(TARGET ${tgt})
      set_target_properties(${tgt} PROPERTIES BUILD_RPATH "@loader_path/../Frameworks" INSTALL_RPATH "@loader_path/../Frameworks")
      add_custom_command(TARGET ${tgt} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_BUNDLE_CONTENT_DIR:${tgt}>/Frameworks"
        COMMAND ${CMAKE_COMMAND} -E copy "${ORT_LIB_REAL}" "$<TARGET_BUNDLE_CONTENT_DIR:${tgt}>/Frameworks/"
        COMMAND codesign --force --sign - "$<TARGET_BUNDLE_DIR:${tgt}>" || true)
    endif()
  endforeach()
endif()

# Headless integration test (no host/GUI needed): `ctest` or run directly.
if(VOXFORGE_BUILD_TESTS)
  add_executable(VoxForgeHeadlessTest Tests/HeadlessTest.cpp)
  target_link_libraries(VoxForgeHeadlessTest PRIVATE VoxForge voxforge_core)
  # Compile against exactly the same JUCE configuration as the plugin's shared code.
  target_include_directories(VoxForgeHeadlessTest PRIVATE $<TARGET_PROPERTY:VoxForge,INCLUDE_DIRECTORIES>)
  target_compile_definitions(VoxForgeHeadlessTest PRIVATE $<TARGET_PROPERTY:VoxForge,COMPILE_DEFINITIONS>)
  add_test(NAME plugin_headless COMMAND VoxForgeHeadlessTest)
endif()
