find_program(FUSEPASS_LIT NAMES lit llvm-lit
  HINTS "${LLVM_TOOLS_BINARY_DIR}"
  DOC "lit test runner (for example from pip install lit)")
if(NOT FUSEPASS_LIT)
  message(WARNING "lit not found: check-fusepass will fail. Run pip install lit or set FUSEPASS_LIT.")
endif()

if(FUSEPASS_ENABLE_PLUGIN)
  set(FUSEPASS_PLUGIN_PATH
      "$<TARGET_FILE:FusePassPlugin>")
endif()

# lit reads this file at test time, so generator expressions are resolved in a
# second step after the normal @VAR@ substitution.
configure_file(lit.site.cfg.py.in ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py.configured @ONLY)
file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  INPUT ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py.configured)

set(FUSEPASS_TEST_DEPENDS fusepass-opt)
if(FUSEPASS_ENABLE_PLUGIN)
  list(APPEND FUSEPASS_TEST_DEPENDS FusePassPlugin)
endif()

add_custom_target(check-fusepass
  COMMAND ${FUSEPASS_LIT} -sv ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${FUSEPASS_TEST_DEPENDS}
  COMMENT "Running the FusePass lit tests"
  USES_TERMINAL
  )
