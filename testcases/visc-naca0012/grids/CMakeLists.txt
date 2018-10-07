
set(MESHBASE "NACA0012_lam_hybrid_")
foreach(imesh 1)
	configure_file(${MESHBASE}${imesh}.msh ${MESHBASE}${imesh}.msh COPYONLY)
endforeach(imesh)
