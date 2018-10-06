
configure_file(laminar-implicit.ctrl laminar-implicit.ctrl)
configure_file(opts.solverc opts.solverc COPYONLY)
configure_file(mgopts.solverc mgopts.solverc COPYONLY)

add_subdirectory(grids)
