/** \page physicspage
 *
 * Non-dimensionalization
 * ======================
 * The equations to be discretized have been non-dimensionalized. The reference variables are
 * - free-stream density
 * - free-stream velocity
 * - free-stream pressure
 * - free-stream molecular viscosity.
 *
 * This is as given according to section 4.14.2 of \cite{matatsuka}. The user input needed in this case
 * is free-stream Mach number, free-stream Reynolds number and free-stream temperature (the latter is only
 * needed for post-processing).
 */
