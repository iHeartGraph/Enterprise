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

#include "graph.h"
#include "bfs_gpu_opt.cuh"
#include <sstream>
#include <iostream>
#include <fstream>

int main(int args, char *argv[]) {
	typedef int vertex_t;
	typedef int index_t;
	
	std::cout<<"Input format: ./exe beg_pos(binary, signed long) csr_file(binary, signed long)\n";
	
	if(args != 3)
	{
		std::cout<<"Wrong input\n";
		return -1;
	}

	const index_t gpu_id 	= 0;
	graph<long, long, double, vertex_t, index_t, double> *ginst 
	= new graph<long, long, double, vertex_t, index_t, double>
		(argv[1],argv[2],NULL);
	
	/*Generate non-redundant non-orphan source list*/
	vertex_t *src_list=new vertex_t[64];
	for(long i=0;i<64;i++)
	{
		vertex_t src=rand()%ginst->vert_count;

		//non-orphan
		if(ginst->beg_pos[src+1]-ginst->beg_pos[src] !=0)
		{
			bool isNew=true;

			//non-redundant
			for(long j=0;j<i;j++)
			{
				if(src==src_list[j])
				{
					isNew=false;
					break;
				}
			}

			if(isNew) src_list[i]=src;
		}
	}

	bfs_gpu_coalescing_mem<vertex_t,index_t>(
		src_list,
		ginst->beg_pos,
		ginst->csr,
		ginst->vert_count,
		ginst->edge_count,
		gpu_id);
	return 0;
}
