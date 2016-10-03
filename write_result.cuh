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
#include <stdlib.h>
#include <time.h>
#include <sstream>
template<typename index_t, typename vertex_t>
int graph<index_t, vertex_t>::
write_result()
{
	std::stringstream ss;
	srand(time(NULL));
	std::ofstream result_file;
	std::string file_str="bfs_result.";
	ss<<rand()%8959;
	file_str.append(ss.str());
	file_str.append(".log");
	result_file.open(file_str.c_str());
	
	for(index_t i=0;;i++)
	{
		index_t counter=0;
		result_file<<"Level "<<i<<"(remaining  vertices) :";
		for(index_t j=0;j<num_ver;j++)
		{
			if(vertex_list[j]->depth==i)
			{
				counter==0 ? :result_file<<",";
				result_file<<vertex_list[j]->src_ver;
				counter++;
			}
		}
		result_file<<"----------Total: "<<counter;
		if(!counter) 
			break;
		result_file<<"\n";
	}

	result_file.close();
	
	return 0;
}
