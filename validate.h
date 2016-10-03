/*
 * Copyright 2016 The George Washington University
 * Written by Hang Liu 
 * Directed by Prof. Howie Huang
 *
 * https://www.seas.gwu.edu/~howie/
 * Contact: iheartgraph@gmail.com
 *
 * 
 * Please cite the following paper:
 * 
 * Hang Liu and H. Howie Huang. 2015. Enterprise: breadth-first graph traversal on GPUs. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '15). ACM, New York, NY, USA, Article 68 , 12 pages. DOI: http://dx.doi.org/10.1145/2807591.2807594
 
 *
 * This file is part of Enterprise.
 *
 * Enterprise is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Enterprise is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Enterprise.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "comm.h"
template < 	typename index_t,
			typename vertex_t,
			typename depth_t> 
index_t validate(	depth_t *depth_h, 
					index_t *beg_pos,
					vertex_t *csr,
					index_t	num_ver)
{
	
	//divide the adjacency vertices of this vertex into two parts
	//-parent part: all the vertices connected to the parent part 
	//				should have smaller or equal to base_depth
	//-children part: all vertices connected to the children part
	//				should have larger or equal to base_depth

	//depth difference should be no more than 1
	
	depth_t 	base_depth;
	vertex_t 	base_v;
	index_t		test = 0;
	depth_t 	conn_depth;
	vertex_t	conn_v;
	
	srand(time(NULL));
	while(test < VALIDATE_TIMES)
	{
		base_v = rand()%num_ver;
		while(depth_h[base_v] == INFTY)
			base_v	= rand()%num_ver;
		
		test ++;
		base_depth	= depth_h[base_v];
		
		for(index_t i = beg_pos[base_v]; i< beg_pos[base_v+1]; i++)
		{
			conn_v		= csr[i];
			conn_depth 	= depth_h[conn_v];
			if(conn_depth > base_depth + 1 ||
				conn_depth < base_depth  -1)
				return -1;
			
			if(conn_depth < base_depth)
			{
				for(index_t j = beg_pos[conn_v];j < beg_pos[conn_v+1]; j++)
				{
					if(depth_h[csr[j]]	> base_depth)
						return -2;
				}
			}

			if(conn_depth > base_depth)
			{
				for(index_t j = beg_pos[conn_v];j < beg_pos[conn_v+1]; j++)
				{
					if(depth_h[csr[j]]
							< base_depth)
						return -3;
				}
			}

			if(conn_depth == base_depth)
			{
				for(index_t j = beg_pos[conn_v];j < beg_pos[conn_v+1]; j++)
				{
					if(	depth_h[csr[j]]> base_depth + 1 ||
						depth_h[csr[j]]	< base_depth -1)
						return -4;
				}
			}
		}
	}
	return 0;
}


template< 	typename vertex_t,
			typename index_t,
			typename depth_t>
void report(	index_t &agg_tr_edges,
				index_t &agg_tr_v,
				index_t *beg_pos,
				depth_t	*depth_h,
				index_t	num_ver)
{
	agg_tr_edges	= 0;
	agg_tr_v		= 0;
	for(index_t i = 0; i< num_ver; i++)
	{
		if(depth_h[i] != INFTY)
		{
			agg_tr_v ++;
			agg_tr_edges += beg_pos[i+1]-beg_pos[i];
		}
	}


}
