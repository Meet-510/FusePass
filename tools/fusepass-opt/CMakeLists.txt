add_llvm_executable(fusepass-opt fusepass-opt.cpp)
llvm_update_compile_flags(fusepass-opt)
target_link_libraries(fusepass-opt PRIVATE
  FusePassTransforms
  MLIROptLib
  MLIRSCFDialect
  )
