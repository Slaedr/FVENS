
add_executable(e_testparse testparse.cpp)
target_link_libraries(e_testparse fvens_base)

add_test(NAME Utils_ParseControlInfo WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} e_testparse
  ${CMAKE_CURRENT_SOURCE_DIR}/inv-explicit.ctrl
  --exact_solution_file ${CMAKE_CURRENT_SOURCE_DIR}/inv-explicit.testdata)

add_executable(e_test_petsc_except test_petsc_exceptions.cpp)
target_link_libraries(e_test_petsc_except fvens_base)

add_test(NAME Utils_TestPetscException WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} e_test_petsc_except)
