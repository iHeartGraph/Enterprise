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
#include "graph.h"
template <typename vertex_t, typename index_t, typename depth_t>
struct allocator{
inline static __host__ void alloc_array
(
	depth_t*	&depth_d,	//vertex level
	vertex_t* 	&adj_list_d,//vertex adj ver
	index_t* 	&adj_card_d,//each adj ver len
	index_t* 	&strt_pos_d,//adj_list_d strt pos
	vertex_t* 	&ex_q_sml_d,//+--------------------
	vertex_t* 	&ex_q_mid_d,//|
	vertex_t* 	&ex_q_lrg_d,//|-------------------+
	index_t* 	&ex_cat_sml_sz,//|USED FOR CLASSIFIC|
	index_t* 	&ex_cat_mid_sz,//|ATION OF CLASSIFYI|
	index_t* 	&ex_cat_lrg_sz,//|NG THE EXPANSION Q|
	index_t* 	&ex_cat_sml_off,//|UEUE-------------+
	index_t* 	&ex_cat_mid_off,//|
	index_t* 	&ex_cat_lrg_off,//+-----------------
	vertex_t*	&ex_cat_sml_d,//each thd obt ex_q 
	vertex_t*	&ex_cat_mid_d,//each thd obt ex_q 
	vertex_t*	&ex_cat_lrg_d,//each thd obt ex_q 
	index_t*	&tr_edges_c_d,
	index_t*	&tr_edges_c_h,
	index_t* beg_pos,
	index_t* csr,
	index_t vert_count,
	index_t edge_count,
	cudaStream_t* 	&stream,
	const index_t	bin_sz
)
{
	//used for 	-strt_pos_d
	//			-adj_card_d
	//			-adj_list_d
	long cpu_bytes	= 0;
	long gpu_bytes	= 0;

	index_t *temp	= new index_t[vert_count];
	cpu_bytes		+= (sizeof(index_t)*vert_count);

	const index_t	cat_sz	= bin_sz*THDS_NUM*BLKS_NUM;
	
	//+----------------------------
	//|ADDED FOR CLASSIFICATION
	//+----------------------------
	H_ERR(cudaMalloc((void **)&ex_cat_sml_d,	sizeof(vertex_t)*cat_sz));
	H_ERR(cudaMalloc((void **)&ex_cat_mid_d,	sizeof(vertex_t)*cat_sz));
	H_ERR(cudaMalloc((void **)&ex_cat_lrg_d,	sizeof(vertex_t)*cat_sz));
	gpu_bytes		+= (3*sizeof(vertex_t)*cat_sz);
	
	H_ERR(cudaMalloc((void **)&ex_q_sml_d,	sizeof(vertex_t)*vert_count));
	H_ERR(cudaMalloc((void **)&ex_q_mid_d,	sizeof(vertex_t)*vert_count));
	H_ERR(cudaMalloc((void **)&ex_q_lrg_d,	sizeof(vertex_t)*vert_count));
	gpu_bytes		+= (sizeof(vertex_t)*vert_count*3);
	
	H_ERR(cudaBindTexture(0, tex_sml_exq, 	
						ex_q_sml_d, sizeof(vertex_t)*vert_count));
	H_ERR(cudaBindTexture(0, tex_mid_exq, 	
						ex_q_mid_d, sizeof(vertex_t)*vert_count));
	H_ERR(cudaBindTexture(0, tex_lrg_exq, 	
						ex_q_lrg_d, sizeof(vertex_t)*vert_count));
	
	const index_t off_sz	= THDS_NUM*BLKS_NUM;
	H_ERR(cudaMalloc((void **)&ex_cat_sml_off, sizeof(index_t)*off_sz));
	H_ERR(cudaMalloc((void **)&ex_cat_mid_off, sizeof(index_t)*off_sz));
	H_ERR(cudaMalloc((void **)&ex_cat_lrg_off, sizeof(index_t)*off_sz));
	H_ERR(cudaMalloc((void **)&ex_cat_sml_sz, sizeof(index_t)*off_sz));
	H_ERR(cudaMalloc((void **)&ex_cat_mid_sz, sizeof(index_t)*off_sz));
	H_ERR(cudaMalloc((void **)&ex_cat_lrg_sz, sizeof(index_t)*off_sz));
	gpu_bytes		+=	(sizeof(vertex_t)*off_sz*6);
	
	
	H_ERR(cudaMalloc((void **)&depth_d, 		sizeof(depth_t)*vert_count));
	H_ERR(cudaBindTexture(0,tex_depth,depth_d,sizeof(depth_t)*vert_count));

	H_ERR(cudaMalloc((void **)&adj_card_d, 	sizeof(index_t)*vert_count));
	H_ERR(cudaMalloc((void **)&strt_pos_d,	sizeof(index_t)*vert_count));
	gpu_bytes		+= (sizeof(index_t)*vert_count*3);
		
	H_ERR(cudaMemcpy(strt_pos_d, beg_pos,	sizeof(index_t)*vert_count, 
				cudaMemcpyHostToDevice));

	H_ERR(cudaBindTexture(0, tex_strt, strt_pos_d, sizeof(index_t)*vert_count));

	EDGES_C = edge_count;
	H_ERR(cudaMalloc((void **)&adj_list_d,sizeof(vertex_t)*edge_count));
	H_ERR(cudaMemcpy(adj_list_d, 
						csr,sizeof(vertex_t)*edge_count, cudaMemcpyHostToDevice));
	gpu_bytes		+= (sizeof(vertex_t)*edge_count);
	
	
	//////////////////////////
	//std::cout<<"before\n";
	stream = (cudaStream_t *)malloc(sizeof(cudaStream_t)*Q_CARD);
	for(index_t i=0;i<Q_CARD; i++)
		cudaStreamCreate(&(stream[i]));

	for(index_t i=0; i<vert_count; i++)
		temp[i]		= beg_pos[i+1]-beg_pos[i];

	H_ERR(cudaMemcpy(adj_card_d, temp, sizeof(index_t)*vert_count, 
								cudaMemcpyHostToDevice));
	H_ERR(cudaBindTexture(0, tex_card, 	 
						adj_card_d, sizeof(index_t)*vert_count));

	H_ERR(cudaMalloc((void **)&tr_edges_c_d, 
					sizeof(index_t)*BLKS_NUM));
	H_ERR(cudaMallocHost((void **)&tr_edges_c_h, 
					sizeof(index_t)*BLKS_NUM));
	
	
	gpu_bytes		+= (sizeof(index_t)*BLKS_NUM);
	cpu_bytes		+= (sizeof(index_t)*BLKS_NUM);

	delete[] temp;
	cpu_bytes		-= (sizeof(index_t)*vert_count);

	std::cout<<"GPU alloc space: "<<gpu_bytes<<" bytes\n";
	std::cout<<"CPU alloc space: "<<cpu_bytes<<" bytes\n";
}
};
