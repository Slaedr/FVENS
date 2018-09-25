
add_executable(runtest_gradientschemes runtest_gradientschemes.cpp testgradientschemes.cpp)
target_link_libraries(runtest_gradientschemes fvens_base)

add_test(NAME LeastSquaresGradients_OneExact WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} runtest_gradientschemes
  ${CMAKE_CURRENT_SOURCE_DIR}/../common-input/testperiodic.msh LEASTSQUARES 1exact)

add_test(NAME LeastSquaresGradients_OneExact_2 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SEQEXEC} ${SEQTASKS} runtest_gradientschemes
  ${CMAKE_CURRENT_SOURCE_DIR}/../common-input/2dcylinderhybrid.msh LEASTSQUARES 1exact)
