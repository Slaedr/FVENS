#include "adatastructures.hpp"

int perm(int start, int end, int n, int off)
{
#ifdef DEBUG
	if(n > end) { std::cout << "Permutation point error!\n"; return 0; }
#endif
	if(off == 0) return n;

	CircList<int> list(start);
	for(int i = start+1; i <= end; i++)
		list.push(i);

	Node<int>* nn = list.find(n);
	Node<int>* cur = nn;
	for(int i = 0; i < off; i++)
		cur = cur->next;
	return cur->data;
} 
