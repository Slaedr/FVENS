#include <vector>
#include "amesh2dh.hpp"

/** Returns a list of cell indices corresponding to the start of each level.
 * The length of the list is the number of levels.
 */
std::vector<a_int> levelSchedule(const UMesh2dh& m)
{
	// zeroth level starts at cell 0
	std::vector<a_int> levels;
	levels.push_back(0);
	
	a_int icell = 0;
	
	while(icell < m.gnelem()-1)
	{
		std::vector<bool> marked(m.gnelem(), false);

		// mark current cell
		marked[icell] = true;

		// mark all neighbors
		for(int iface = 0; iface < m.gnfael(icell); iface++)
			marked[m.gesuel(icell,iface)] = true;

		/* If the next cell is among marked cells, this level ends at this cell
		 * and the next level starts at the next cell.
		 */
		if(marked[icell+1])
			levels.push_back(icell+1);

		icell++;
	}

	return levels;
}
