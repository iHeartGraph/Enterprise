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

////////////////////////////
//EXTERNAL VARIABLES
///////////////////////////
//__device__ int expand_type_d;


//////////////////////////////////
//This file is mainly about expander
//--------------------------------------------------
//	For expander, it should expand based on ex_queue_d
//and put all the expanded data into inspection
//--------------------------------------------------


__device__ void __sync_warp(int predicate)
{
	while((!__all(predicate)))
	{
		;
	}
}

//This kernel is executed by one thread
template<typename vertex_t, 
		typename index_t,
		typename depth_t>
__global__ void init_expand_sort
(	
	vertex_t src_v,
	depth_t	*depth_d
)
{
	depth_d[src_v]	= 0;	
	in_q_sz_d		= 1;
	error_d 		= 0;
	return ;
}


//+---------------------------
//|for ex_q_sml_d expansion
//+---------------------------
template<typename vertex_t, 
		typename index_t,
		typename depth_t>
__global__ void THD_expand_sort
(	
	depth_t	*depth_d,
	index_t	curr_level,
	const vertex_t* __restrict__ adj_list_d
)
{
	const index_t q_sz		= ex_sml_sz_d;
	const depth_t LEVEL		= curr_level;
	
	const index_t GRNLTY	= blockDim.x * gridDim.x;
	index_t tid				= threadIdx.x+blockIdx.x*blockDim.x;

	//used for prefetching
	vertex_t 	ex_ver;
	index_t 	card_curr, card_next;
	index_t 	strt_pos_curr, strt_pos_next;
	vertex_t 	aq_ver_curr, aq_ver_next;
	depth_t 	adj_depth_curr, adj_depth_next;
	__shared__ index_t	hub_cache[HUB_SZ];
	__shared__ depth_t	hub_depth[HUB_SZ];
	
	index_t	 cache_ptr	= threadIdx.x;
	
	while(cache_ptr < HUB_SZ)
	{
		hub_cache[cache_ptr] 	= hub_vert[cache_ptr];
		hub_depth[cache_ptr]	= INFTY;

		cache_ptr += blockDim.x;
	}

	__syncthreads();

	//prefetching
	if (tid < q_sz)
	{
		ex_ver			= tex1Dfetch(tex_sml_exq, tid);
		card_curr		= tex1Dfetch(tex_card, ex_ver);
		strt_pos_curr	= tex1Dfetch(tex_strt, ex_ver);
	}

	while(tid<q_sz)
	{
		tid 	  	   += GRNLTY;
		if(tid < q_sz)
		{
			ex_ver			= tex1Dfetch(tex_sml_exq, tid);
			card_next		= tex1Dfetch(tex_card, ex_ver);
			strt_pos_next	= tex1Dfetch(tex_strt, ex_ver);
		}

		index_t lane = strt_pos_curr;
		card_curr	+= strt_pos_curr;

		aq_ver_curr	= adj_list_d[lane];
		cache_ptr	= aq_ver_curr & (HUB_SZ - 1);
		if(aq_ver_curr == hub_cache[cache_ptr])
		{
			adj_depth_curr = LEVEL - 1; 
			hub_depth[cache_ptr] = LEVEL;
		}else{
			adj_depth_curr	= depth_d[aq_ver_curr];
		}
		
		while(lane < card_curr)
		{
			lane++; 
			if(lane < card_curr)
			{
				aq_ver_next	= adj_list_d[lane];
				
				cache_ptr	= aq_ver_next & (HUB_SZ - 1);
				if(aq_ver_next == hub_cache[cache_ptr])
				{
					adj_depth_next = LEVEL - 1; 
					hub_depth[cache_ptr] = LEVEL;
				}else{
					adj_depth_next	= depth_d[aq_ver_next];
				}
			}
			
			//0	unvisited 	0x00
			//1	fontier		0x01
			//2 visited		0x02
			if(adj_depth_curr == INFTY)
				depth_d[aq_ver_curr]= LEVEL;
			
			aq_ver_curr		= aq_ver_next;
			adj_depth_curr	= adj_depth_next;
		}
		
		card_curr 		= card_next;
		strt_pos_curr	= strt_pos_next;
	}
	__syncthreads();
	cache_ptr	= threadIdx.x;

	while(cache_ptr < HUB_SZ)
	{
		//hub_depth should be investigated before depth_d[hub_cache[]]
		//Reason: hub_cache[] maybe blank which leads to out-of-bound 
		//			depth_d transaction
		if((hub_depth[cache_ptr] == LEVEL)
			&& (depth_d[hub_cache[cache_ptr]] == INFTY))
			depth_d[hub_cache[cache_ptr]]= LEVEL;

		cache_ptr += blockDim.x;
	}
}

//+------------------------------
//|ex_q_mid_d expansion
//+------------------------------
template<typename vertex_t, 
		typename index_t,
		typename depth_t>
__global__ void WAP_expand_sort
(	
	depth_t		*depth_d,
	index_t		curr_level,
	const vertex_t* __restrict__ adj_list_d
)
{
	const index_t q_sz		= ex_mid_sz_d;
	const depth_t LEVEL		= curr_level;

	const index_t vec_sz	= ((THDS_NUM>=32)? 32:1);
	const index_t tid		= threadIdx.x+blockIdx.x*blockDim.x;
	const index_t lane_s	= tid & (vec_sz-1);
	const index_t GRNLTY	= (blockDim.x * gridDim.x)/vec_sz;
	index_t	vec_id			= tid/vec_sz;

	//used for prefetching
	vertex_t 	ex_ver;
	index_t 	card_curr, card_next;
	index_t 	strt_pos_curr, strt_pos_next;
	vertex_t 	aq_ver_curr, aq_ver_next;
	depth_t 	adj_depth_curr, adj_depth_next;
	
	__shared__ index_t		hub_cache[HUB_SZ];
	__shared__ depth_t	hub_depth[HUB_SZ];

	index_t	 cache_ptr	= threadIdx.x;
	while(cache_ptr < HUB_SZ)
	{
		hub_cache[cache_ptr] 	= hub_vert[cache_ptr];
		hub_depth[cache_ptr]	= INFTY;
		cache_ptr += blockDim.x;
	}
	__syncthreads();
	
	//prefetching
	if (vec_id < q_sz)
	{
		ex_ver			= tex1Dfetch(tex_mid_exq, vec_id);
		card_curr		= tex1Dfetch(tex_card, ex_ver);
		strt_pos_curr	= tex1Dfetch(tex_strt, ex_ver);
	}

	while(vec_id < q_sz)
	{
		vec_id 	  	   += GRNLTY;
		if(vec_id < q_sz)
		{
			ex_ver			= tex1Dfetch(tex_mid_exq, vec_id);
			card_next		= tex1Dfetch(tex_card, ex_ver);
			strt_pos_next	= tex1Dfetch(tex_strt, ex_ver);
		}
	
		index_t lane	= lane_s + strt_pos_curr;
		aq_ver_curr		= adj_list_d[lane];

		cache_ptr		= aq_ver_curr & (HUB_SZ -1);
		if(aq_ver_curr == hub_cache[cache_ptr])
		{
			adj_depth_curr = LEVEL - 1; 
			hub_depth[cache_ptr] = LEVEL;
		}
		else{
			adj_depth_curr	= depth_d[aq_ver_curr];
		}
		card_curr	   += strt_pos_curr;
		
		while(lane < card_curr)
		{
			lane		+= vec_sz; 
			if(lane < card_curr)
			{
				aq_ver_next	= adj_list_d[lane];
				
				cache_ptr	= aq_ver_next & (HUB_SZ - 1);
				if(aq_ver_next == hub_cache[cache_ptr])
				{
					adj_depth_next = LEVEL - 1; 
					hub_depth[cache_ptr] = LEVEL;
				}else{
					adj_depth_next	= depth_d[aq_ver_next];
				}
			}
			
			//0	unvisited 	0x00
			//1	fontier		0x01
			//2 visited		0x02
			if(adj_depth_curr == INFTY)
				depth_d[aq_ver_curr]= LEVEL;
			
			aq_ver_curr		= aq_ver_next;
			adj_depth_curr	= adj_depth_next;
		}
		__sync_warp(1);
		
		card_curr 		= card_next;
		strt_pos_curr	= strt_pos_next;
	}
	__syncthreads();

	cache_ptr	= threadIdx.x;

	while(cache_ptr < HUB_SZ)
	{
		//hub_depth should be investigated before depth_d[hub_cache[]]
		//Reason: hub_cache[] maybe blank which leads to out-of-bound 
		//			depth_d transaction
		if((hub_depth[cache_ptr] == LEVEL)
			&& (depth_d[hub_cache[cache_ptr]] == INFTY))
			depth_d[hub_cache[cache_ptr]]= LEVEL;

		cache_ptr += blockDim.x;
	}
}

template<typename vertex_t, 
		typename index_t,
		typename depth_t>
__global__ void CTA_expand_sort
(	
	depth_t			*depth_d,
	index_t			curr_level,
	const vertex_t* __restrict__ adj_list_d
)
{
	const index_t	q_sz = ex_lrg_sz_d;
	index_t	vec_id	= blockIdx.x;
	const depth_t LEVEL		= curr_level;

	//used for prefetching
	vertex_t 	ex_ver;
	index_t 	card_curr, card_next;
	index_t 	strt_pos_curr, strt_pos_next;
	vertex_t 	aq_ver_curr, aq_ver_next;
	depth_t		adj_depth_curr, adj_depth_next;
	
	__shared__ index_t	hub_cache[HUB_SZ];
	__shared__ depth_t	hub_depth[HUB_SZ];
	
	index_t	 cache_ptr	= threadIdx.x;
	while(cache_ptr < HUB_SZ)
	{
		hub_cache[cache_ptr] 	= hub_vert[cache_ptr];
		hub_depth[cache_ptr]	= INFTY;
		cache_ptr += blockDim.x;
	}
	__syncthreads();

	//prefetching
	if (vec_id<q_sz)
	{
		ex_ver			= tex1Dfetch(tex_lrg_exq, vec_id);
		card_curr		= tex1Dfetch(tex_card, ex_ver);
		strt_pos_curr	= tex1Dfetch(tex_strt, ex_ver);
	}

	while(vec_id<q_sz)
	{
		vec_id += gridDim.x;
		
		if(vec_id < q_sz)
		{
			ex_ver			= tex1Dfetch(tex_lrg_exq, vec_id);
			card_next		= tex1Dfetch(tex_card, ex_ver);
			strt_pos_next	= tex1Dfetch(tex_strt, ex_ver);
		}
		
		index_t lane	= threadIdx.x + strt_pos_curr;

		aq_ver_curr	= adj_list_d[lane];
		cache_ptr	= aq_ver_curr & (HUB_SZ - 1);
		if(aq_ver_curr == hub_cache[cache_ptr])
		{
			adj_depth_curr 			= LEVEL - 1; 
			hub_depth[cache_ptr] 	= LEVEL;
		}else{
			adj_depth_curr	= depth_d[aq_ver_curr];
		}
		
		card_curr	   += strt_pos_curr;
		while(lane < card_curr)
		{
			//-reimburse lane for prefetch
			//-check lane and prefetch for next
			//	iteration
			lane	+= blockDim.x;
			if(lane < card_curr)
			{
				aq_ver_next	= adj_list_d[lane];
				
				cache_ptr	= aq_ver_next & (HUB_SZ - 1);
				if(aq_ver_next == hub_cache[cache_ptr])
				{
					adj_depth_next = LEVEL - 1; 
					hub_depth[cache_ptr] = LEVEL;
				}else{
					adj_depth_next	= depth_d[aq_ver_next];
				}
			}
			
			//0	unvisited 	0x00
			//1	fontier		0x01
			//2 visited		0x02
			if(adj_depth_curr == INFTY)
				depth_d[aq_ver_curr]= LEVEL;
			
			aq_ver_curr	= aq_ver_next;
			adj_depth_curr= adj_depth_next;
		}
		__syncthreads();
		
		card_curr 		= card_next;
		strt_pos_curr	= strt_pos_next;
	}
	__syncthreads();
	
	cache_ptr	= threadIdx.x;

	while(cache_ptr < HUB_SZ)
	{
		//hub_depth should be investigated before depth_d[hub_cache[]]
		//Reason: hub_cache[] maybe blank which leads to out-of-bound 
		//			depth_d transaction
		if((hub_depth[cache_ptr] == LEVEL)
			&& (depth_d[hub_cache[cache_ptr]] == INFTY))
			depth_d[hub_cache[cache_ptr]]= LEVEL;

		cache_ptr += blockDim.x;
	}
}

//+-------------------------------------------------------------------
//|BOTTOM UP EXPANSION FUNCTIONS
//+--------------------------------------------------------------------
//+---------------------------
//|for ex_q_sml_d expansion
//+---------------------------
template<typename vertex_t, 
		typename index_t,
		typename depth_t>
__global__ void THD_bu_expand_sort
(	
	depth_t			*depth_d,
	index_t			curr_level,
	const vertex_t* __restrict__ adj_list_d
)
{
	const index_t q_sz		= ex_sml_sz_d;
	
	const index_t GRNLTY	= blockDim.x * gridDim.x;
	index_t tid				= threadIdx.x+blockIdx.x*blockDim.x;
	const depth_t LEVEL		= curr_level;
	const depth_t LST_LEVEL	= LEVEL - 1;

	//used for prefetching
	vertex_t 	ex_ver_curr, ex_ver_next;
	index_t 	card_curr, card_next;
	index_t 	strt_pos_curr, strt_pos_next;
	vertex_t 	aq_ver_curr, aq_ver_next;
	depth_t 	adj_depth_curr, adj_depth_next;
	__shared__ index_t	hub_cache[HUB_BU_SZ];
	
	index_t	 cache_ptr	= threadIdx.x;
	while(cache_ptr < HUB_BU_SZ)
	{
		hub_cache[cache_ptr] 	= hub_vert[cache_ptr];
		cache_ptr += blockDim.x;
	}
	__syncthreads();
	
	//prefetching
	if (tid < q_sz)
	{
		ex_ver_curr		= tex1Dfetch(tex_sml_exq, tid);
		card_curr		= tex1Dfetch(tex_card, ex_ver_curr);
		strt_pos_curr	= tex1Dfetch(tex_strt, ex_ver_curr);
	}

	while(tid<q_sz)
	{
		tid 	  	   += GRNLTY;
		if(tid < q_sz)
		{
			ex_ver_next		= tex1Dfetch(tex_sml_exq, tid);
			card_next		= tex1Dfetch(tex_card, ex_ver_next);
			strt_pos_next	= tex1Dfetch(tex_strt, ex_ver_next);
		}

		index_t lane = strt_pos_curr;
		card_curr	+= strt_pos_curr;

		aq_ver_curr	= adj_list_d[lane];
		cache_ptr	= aq_ver_curr & (HUB_BU_SZ - 1);
		if(aq_ver_curr == hub_cache[cache_ptr])
		{
			depth_d[ex_ver_curr] = LEVEL;
				
			ex_ver_curr		= ex_ver_next;
			card_curr 		= card_next;
			strt_pos_curr	= strt_pos_next;
			continue;
		}else{
			adj_depth_curr	= depth_d[aq_ver_curr];
		}

		while(lane < card_curr)
		{
			lane++;
			if(lane < card_curr)
			{
				aq_ver_next	= adj_list_d[lane];
				
				cache_ptr = aq_ver_next & (HUB_BU_SZ - 1);
				if(aq_ver_next == hub_cache[cache_ptr])
				{
					depth_d[ex_ver_curr] = LEVEL;
					break;
				}else{
					adj_depth_next	= depth_d[aq_ver_next];
				}
			}
			
			//0	unvisited 	0x00
			//1	fontier		0x01
			//2 visited		0x02
			if(adj_depth_curr == LST_LEVEL)
			{
				depth_d[ex_ver_curr] = LEVEL;
				break;
			}
			aq_ver_curr		= aq_ver_next;
			adj_depth_curr	= adj_depth_next;
		}
		
		ex_ver_curr		= ex_ver_next;
		card_curr 		= card_next;
		strt_pos_curr	= strt_pos_next;
	}
}

//+------------------------------
//|ex_q_mid_d expansion
//+------------------------------
template<typename vertex_t, 
		typename index_t,
		typename depth_t>
__global__ void WAP_bu_expand_sort
(	
	depth_t			*depth_d,
	index_t			curr_level,
	const vertex_t* __restrict__ adj_list_d
)

{
	const index_t q_sz		= ex_mid_sz_d;

	const index_t vec_sz	= ((THDS_NUM>=32)? 32:1);
	const index_t tid		= threadIdx.x+blockIdx.x*blockDim.x;
	const index_t lane_s	= tid & (vec_sz-1);
	const index_t GRNLTY	= (blockDim.x * gridDim.x)/vec_sz;
	index_t	vec_id			= tid/vec_sz;
	const depth_t LEVEL		= curr_level;
	const depth_t LST_LEVEL	= LEVEL - 1;

	//used for prefetching
	vertex_t 	ex_ver_curr, ex_ver_next;
	index_t 	card_curr, card_next;
	index_t 	card_curr_revised, card_next_revised;
	index_t 	strt_pos_curr, strt_pos_next;
	vertex_t 	aq_ver_curr, aq_ver_next;
	depth_t 	adj_depth_curr, adj_depth_next;

	__shared__ index_t	hub_cache[HUB_BU_SZ];
	
	index_t	 cache_ptr	= threadIdx.x;
	while(cache_ptr < HUB_BU_SZ)
	{
		hub_cache[cache_ptr] 	= hub_vert[cache_ptr];
		cache_ptr += blockDim.x;
	}
	__syncthreads();
	
	//prefetching
	if (vec_id < q_sz)
	{
		ex_ver_curr		= tex1Dfetch(tex_mid_exq, vec_id);
		card_curr		= tex1Dfetch(tex_card, ex_ver_curr);
		strt_pos_curr	= tex1Dfetch(tex_strt, ex_ver_curr);
		if(card_curr%vec_sz)
		{
			card_curr_revised = (((card_curr>>5)+1)<<5);
		}else{
			card_curr_revised = card_curr;
		}
	}

	while(vec_id < q_sz)
	{
		vec_id 	  	   += GRNLTY;
		if(vec_id < q_sz)
		{
			ex_ver_next		= tex1Dfetch(tex_mid_exq, vec_id);
			card_next		= tex1Dfetch(tex_card, ex_ver_next);
			strt_pos_next	= tex1Dfetch(tex_strt, ex_ver_next);
			if(card_next%vec_sz)
			{
				card_next_revised = (((card_next>>5)+1)<<5);
			}else{
				card_next_revised = card_next;
			}
		}
	
		index_t lane		= lane_s + strt_pos_curr;
		card_curr			+= strt_pos_curr;
		card_curr_revised	+= strt_pos_curr;

		//cardinality of all vertices in wap_queue 
		//is larger than 32
		aq_ver_curr		= adj_list_d[lane];

		cache_ptr	= aq_ver_curr & (HUB_BU_SZ - 1);
		__sync_warp(1);
		if(__any(aq_ver_curr== hub_cache[cache_ptr]))
		{
			if(!lane_s){
				depth_d[ex_ver_curr] = LEVEL;
			}
			__sync_warp(1);
			
			ex_ver_curr		= ex_ver_next;
			card_curr 		= card_next;
			card_curr_revised= card_next_revised;
			strt_pos_curr	= strt_pos_next;
				
			continue;
		}else{
			adj_depth_curr	= depth_d[aq_ver_curr];
		}

		while(lane < card_curr_revised)
		{
			//prefetching for the next iteration
			lane		+= vec_sz; 
			if(lane < card_curr){//0~33 vertices, second round 32, 33
				aq_ver_next	= adj_list_d[lane];
				cache_ptr = aq_ver_next & (HUB_BU_SZ - 1);
			}else{//0~33 vertices, second round 34, ..., 63 vertices
				aq_ver_next	= V_NON_INC;
				cache_ptr = 0;
			}
			
			if(__any(aq_ver_next == hub_cache[cache_ptr]))
			{
				if(!lane_s){
					SET_VIS(depth_d[ex_ver_curr]);
					depth_d[ex_ver_curr] = LEVEL;
				}
				break;
			}else{
				if(lane < card_curr) adj_depth_next 
										= depth_d[aq_ver_next];
				else	adj_depth_next	= 0;
			}

			//0	unvisited 	0x00
			//1	fontier		0x01
			//2 visited		0x02
			__sync_warp(1);
			
			if(__any(adj_depth_curr == LST_LEVEL))
			{
				if(!lane_s){
					depth_d[ex_ver_curr] = LEVEL;
				}
				
				break;
			}
			
			aq_ver_curr		= aq_ver_next;
			adj_depth_curr	= adj_depth_next;
		}
		__sync_warp(1);
		
		ex_ver_curr		= ex_ver_next;
		card_curr 		= card_next;
		card_curr_revised= card_next_revised;
		strt_pos_curr	= strt_pos_next;
	}
}

template<typename vertex_t, 
		typename index_t,
		typename depth_t>
__global__ void CTA_bu_expand_sort
(	
	depth_t			*depth_d,
	index_t			curr_level,
	const vertex_t* __restrict__ adj_list_d
)
{
	const index_t	q_sz = ex_lrg_sz_d;
	
	index_t	vec_id 		= blockIdx.x;
	vertex_t 	ex_ver_curr, ex_ver_next;
	index_t 	card_curr, card_next;
	index_t 	card_curr_revised, card_next_revised;
	index_t 	strt_pos_curr, strt_pos_next;
	vertex_t 	aq_ver_curr, aq_ver_next;
	depth_t		adj_depth_curr, adj_depth_next;
	const depth_t LEVEL		= curr_level;
	const depth_t LST_LEVEL	= LEVEL - 1;

	__shared__ index_t	hub_cache[HUB_BU_SZ];
	
	index_t	 cache_ptr	= threadIdx.x;
	while(cache_ptr < HUB_BU_SZ)
	{
		hub_cache[cache_ptr] 	= hub_vert[cache_ptr];
		cache_ptr += blockDim.x;
	}
	__syncthreads();
	
	//prefetching
	if (vec_id < q_sz)
	{
		ex_ver_curr		= tex1Dfetch(tex_lrg_exq, vec_id);
		card_curr		= tex1Dfetch(tex_card, ex_ver_curr);
		strt_pos_curr	= tex1Dfetch(tex_strt, ex_ver_curr);
		if(card_curr%blockDim.x)
		{
			card_curr_revised = card_curr + blockDim.x 
								- (card_curr%blockDim.x); 
		}else{
			card_curr_revised = card_curr;
		}
	}

	while(vec_id < q_sz)
	{
		vec_id += gridDim.x;
		
		if(vec_id<q_sz)
		{
			ex_ver_next		= tex1Dfetch(tex_lrg_exq, vec_id);
			card_next		= tex1Dfetch(tex_card, ex_ver_next);
			strt_pos_next	= tex1Dfetch(tex_strt, ex_ver_next);
			if(card_next%blockDim.x)
			{
				card_next_revised = card_next+blockDim.x 
									- (card_next%blockDim.x); 
			}else{
				card_next_revised = card_next;
			}
		}
		
		index_t lane		= threadIdx.x + strt_pos_curr;
		card_curr			+= strt_pos_curr;
		card_curr_revised	+= strt_pos_curr;

		//cardinality of all vertices in cta_queue
		//is larger than num_threads_in_block
		aq_ver_curr		= adj_list_d[lane];

		cache_ptr	= aq_ver_curr & (HUB_BU_SZ - 1);
		__syncthreads();
		if(__syncthreads_or(aq_ver_curr== hub_cache[cache_ptr]))
		{
			if(!threadIdx.x) depth_d[ex_ver_curr] = LEVEL;

			__syncthreads();
			
			ex_ver_curr	= ex_ver_next;
			card_curr 	= card_next;
			card_curr_revised 	= card_next_revised;
			strt_pos_curr= strt_pos_next;
			continue;
		}else{
			adj_depth_curr	= depth_d[aq_ver_curr];
		}
		
		while(lane < card_curr_revised)
		{
			//-reimburse lane for prefetch
			//-check lane and prefetch for next
			//	iteration
			lane	+= blockDim.x;
			if(lane < card_curr){//0~257 vertices,second round 256, 257
				aq_ver_next	= adj_list_d[lane];
				cache_ptr = aq_ver_next & (HUB_BU_SZ - 1);
			}else{//0~257 vertices, second round 258, ..., 511 vertices
				aq_ver_next	= V_NON_INC;
				cache_ptr = 0;
			}
			
			__syncthreads();
			if(__syncthreads_or(aq_ver_next == hub_cache[cache_ptr]))
			{
				if(!threadIdx.x) depth_d[ex_ver_curr] = LEVEL;

				break;
			}else{
				if(lane < card_curr) adj_depth_next 
										= depth_d[aq_ver_next];
				else	adj_depth_next	= 0;
			}

			//0	unvisited 	0x00
			//1	fontier		0x01
			//2 visited		0x02
			__syncthreads();
			if(__syncthreads_or(adj_depth_curr == LST_LEVEL))
			{
				if(!threadIdx.x) depth_d[ex_ver_curr] = LEVEL;
				break;
			}
			aq_ver_curr	= aq_ver_next;
			adj_depth_curr= adj_depth_next;
		}
		__syncthreads();
		
		ex_ver_curr	= ex_ver_next;
		card_curr 	= card_next;
		card_curr_revised 	= card_next_revised;
		strt_pos_curr= strt_pos_next;
	}
}


//+------------------------------------------------------
//|Following presents the two types of expanders
//|	1. scan_expander
//| 2. sort_expander
//| scan and sort are only used for clarify how they store
//|their expanded results into depth_d
//+------------------------------------------------------
//|Both of them exploit warp_expander and CTA_expander
//|for expanding different type of ex_queue_d candidates
//+------------------------------------------------------
/*
 Expand ex_queue_d, put all expanded data into depth_d
 by scan offset and adds up
 */

//+----------------------
//|CLFY_EXPAND_SORT
//+----------------------
template<typename vertex_t, 
		typename index_t,
		typename depth_t>
void clfy_expand_sort
(	
	depth_t 		*depth_d,
	index_t			curr_level,
	const vertex_t 	*adj_list_d,
	cudaStream_t 	*stream
)
{
	THD_expand_sort<vertex_t, index_t, depth_t>
	<<<BLKS_NUM, THDS_NUM, 0, stream[0]>>>
	(
		depth_d,
		curr_level,
		adj_list_d
	);
//	std::cout<<"td THD ="<<cudaDeviceSynchronize()<<"\n";	
	
	CTA_expand_sort<vertex_t, index_t, depth_t>
	<<<BLKS_NUM, THDS_NUM, 0, stream[2]>>>
	(
		depth_d,
		curr_level,
		adj_list_d
	);
//	std::cout<<"td CTA ="<<cudaDeviceSynchronize()<<"\n";	
	
	WAP_expand_sort<vertex_t, index_t, depth_t>
	<<<BLKS_NUM, THDS_NUM, 0, stream[1]>>>
	(
		depth_d,
		curr_level,
		adj_list_d
	);
//	std::cout<<"td WAP ="<<cudaDeviceSynchronize()<<"\n";	
	
}

//+----------------------
//|CLFY_EXPAND_SORT
//+----------------------
template<typename vertex_t, 
		typename index_t,
		typename depth_t>
void clfy_bu_expand_sort
(	
	depth_t 		*depth_d,
	index_t			curr_level,
	const vertex_t 	*adj_list_d,
	cudaStream_t 	*stream
)
{
	THD_bu_expand_sort<vertex_t, index_t, depth_t>
	<<<BLKS_NUM, THDS_NUM, 0, stream[0]>>>
	(
		depth_d,
		curr_level,
		adj_list_d
	);
	//std::cout<<"bu THD ="<<cudaDeviceSynchronize()<<"\n";	
		
	WAP_bu_expand_sort<vertex_t, index_t, depth_t>
	<<<BLKS_NUM, THDS_NUM, 0, stream[1]>>>
	(
		depth_d,
		curr_level,
		adj_list_d
	);
	//std::cout<<"bu WAP ="<<cudaDeviceSynchronize()<<"\n";	
	
	CTA_bu_expand_sort<vertex_t, index_t, depth_t>
	<<<BLKS_NUM, THDS_NUM, 0, stream[2]>>>
	(
		depth_d,
		curr_level,
		adj_list_d
	);
	//std::cout<<"bu CTA ="<<cudaDeviceSynchronize()<<"\n";	
}

