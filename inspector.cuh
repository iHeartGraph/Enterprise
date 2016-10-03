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
//Dec/15/2013
#include "comm.h"

//+---------------------------------
//|For clfy BASED EX_QUEUE STORAGE
//+---------------------------------
template <typename vertex_t, typename index_t,
			typename depth_t>
__global__ void sort_dist_inspect_clfy
(
	vertex_t	*ex_cat_sml_q, 
	vertex_t	*ex_cat_mid_q, 
	vertex_t	*ex_cat_lrg_q, 
	index_t		*tr_edges_c_d,
	index_t	 	*ex_cat_sml_sz,
	index_t	 	*ex_cat_mid_sz,
	index_t	 	*ex_cat_lrg_sz,
	depth_t 	*depth_d,
	depth_t	 	curr_level,
	index_t	 	num_ver,
	const index_t sml_shed,
	const index_t lrg_shed,
	const index_t bin_sz
)
{
	const index_t TID_ST	= threadIdx.x +blockIdx.x*blockDim.x;
	const index_t NUM_VER	= num_ver;	
	const index_t GRNTY		= blockDim.x*gridDim.x;
	const index_t EX_OFF	= TID_ST*bin_sz;
	index_t tid_next		= TID_ST;
	index_t tid				= TID_ST;
	const depth_t 	LEVEL 	= curr_level;
	
	__shared__ index_t cache[THDS_NUM];
	depth_t		depth_curr, depth_next;
	index_t 	card_curr, card_next;
	
	index_t ex_sml_ct		= 0;
	index_t ex_mid_ct		= 0;
	index_t ex_lrg_ct		= 0;
	index_t card_sum		= 0;

	if(tid < NUM_VER){
		card_curr	= tex1Dfetch(tex_card, tid);
		depth_curr	= tex1Dfetch(tex_depth, tid);
		tid_next	+= GRNTY;
	}
	
	while(tid<NUM_VER){
		if(tid_next < NUM_VER){
			card_next	= tex1Dfetch(tex_card, tid_next);
			depth_next	= tex1Dfetch(tex_depth, tid_next);
		}

		if(depth_curr == LEVEL){
			card_sum 	+= card_curr;
			//---------------
			//!!!HERE IS A FILTER OF CARD =0
			//------------------------------
			if(card_curr<=0)
			{}else if(card_curr < sml_shed)
			{
				ex_cat_sml_q[EX_OFF+ex_sml_ct]=tid;
				ex_sml_ct++;
			}else if(card_curr > lrg_shed){
				ex_cat_lrg_q[EX_OFF+ex_lrg_ct]=tid;
				ex_lrg_ct++;
			}else{
				ex_cat_mid_q[EX_OFF+ex_mid_ct]=tid;
				ex_mid_ct++;
			}
		}
		
		card_curr	= card_next;
		depth_curr	= depth_next;
		tid			= tid_next;
		tid_next	+= GRNTY;
	}
#ifdef ENABLE_CHECKING
	if(	ex_sml_ct > bin_sz ||
		ex_mid_ct > bin_sz ||
		ex_lrg_ct > bin_sz)
		error_d = 1;
#endif

	ex_cat_sml_sz[TID_ST]= ex_sml_ct;
	ex_cat_mid_sz[TID_ST]= ex_mid_ct;
	ex_cat_lrg_sz[TID_ST]= ex_lrg_ct;
	
	if(LEVEL != 0){
		cache[threadIdx.x] = card_sum;
		__syncthreads();
		
		int i = blockDim.x>>1;
		while(i){
			if(threadIdx.x <i)
				cache[threadIdx.x] += cache[threadIdx.x+i];
			__syncthreads();
			i>>=1;
		}

		if(!threadIdx.x)
			tr_edges_c_d[blockIdx.x] = cache[0];
	}
}


template <typename vertex_t, typename index_t,
			typename depth_t>
__global__ void sort_switch_dist_inspect_clfy
(
	vertex_t *ex_cat_sml_q,//each thd obt ex_q 
	vertex_t *ex_cat_mid_q,//each wap obt ex_q 
	vertex_t *ex_cat_lrg_q,//each cta obt ex_q 
	index_t	 *ex_cat_sml_sz,
	index_t	 *ex_cat_mid_sz,
	index_t	 *ex_cat_lrg_sz,
	depth_t *depth_d,
	depth_t	 curr_level,
	index_t	 num_ver,
	const index_t sml_shed,
	const index_t lrg_shed,
	const index_t bin_sz
)
{
	const index_t TID_ST	= threadIdx.x +blockIdx.x*blockDim.x;
	const index_t NUM_VER	= num_ver;	
	const index_t GRNTY		= blockDim.x*gridDim.x;
	const index_t EX_OFF	= TID_ST*bin_sz;
	const depth_t LEVEL 	= curr_level;

	index_t card_curr, card_next;
	depth_t	depth_curr, depth_next;
	
	index_t ex_sml_ct	= 0;
	index_t ex_mid_ct	= 0;
	index_t ex_lrg_ct	= 0;
	index_t hub_ptr;
	index_t tid				= TID_ST;
	
	while(tid < HUB_BU_SZ){
		hub_vert[tid]	= V_INI_HUB;
		hub_card[tid]	= 0;
		tid	+= GRNTY;
	}
	
	//Figure out its own start and end pos
	//We want each thread to inspect a continuous block
	index_t step_sz = NUM_VER/GRNTY;
//	if(step_sz<16){	
		//small problem size
		const index_t REMAINDER = NUM_VER - step_sz * GRNTY;
		if(TID_ST < REMAINDER){
			step_sz ++;
		}
		const index_t strt_pos = step_sz * TID_ST + 
					(TID_ST >= REMAINDER ? REMAINDER:0);
		const index_t end_pos  = strt_pos + step_sz;

		tid		 		 = strt_pos;
		index_t tid_next = strt_pos + 1; 
		if(step_sz){
			card_curr	= tex1Dfetch(tex_card,  strt_pos);
			depth_curr	= tex1Dfetch(tex_depth, strt_pos);
		}
		
		while(tid < end_pos){
			if( tid_next < end_pos){
				card_next	= tex1Dfetch(tex_card, tid_next);
				depth_next	= tex1Dfetch(tex_depth,tid_next);
			}
			
			if(depth_curr == INFTY){
				if(card_curr > 0){
					if(card_curr < sml_shed){
						ex_cat_sml_q[EX_OFF+ex_sml_ct]=tid;
						ex_sml_ct++;
					}else if(card_curr > lrg_shed){
						ex_cat_lrg_q[EX_OFF+ex_lrg_ct]=tid;
						ex_lrg_ct++;
					}else{
						ex_cat_mid_q[EX_OFF+ex_mid_ct]=tid;
						ex_mid_ct++;
					}
				}
			}
			
			//construct hub-cache
			if(depth_curr == LEVEL){
				hub_ptr	= tid & (HUB_BU_SZ - 1);
				if(card_curr > hub_card[hub_ptr]){
					hub_vert[hub_ptr] = tid;
					hub_card[hub_ptr] = card_curr;
				}
			}
			
			card_curr	= card_next;
			depth_curr	= depth_next;
			tid			= tid_next;
			tid_next	++;
		}
//	}else{
//		//big problem size
//		//In problem big, we want each thread to get 16x indices to check.
//		if(NUM_VER - step_sz * GRNTY){
//			//handle error
//		}
//		__shared__ index_t	beg_pos_s[THDS_NUM];
//		__shared__ depth_t	depth_s[THDS_NUM<<4];
//		beg_pos_s[threadIdx.x] = step_sz*TID;
//	
//		const index_t TRIES 		= step_sz>>4;
//		const index_t lane_id		= threadIdx.x%32;
//		const index_t	warp_id		= threadIdx.x>>5;
//		const index_t	thread_off= threadIdx.x<<4;
//
//		index_t tries = 0;
//		index_t	load_ptr;
//		index_t	proc_vert;
//		
//		while(tries < TRIES){
//
//			//warp stride loading
//			for(int i = 0; i< 32; i++){
//				proc_vert=(warp_id<<5)+i+(lane_id>>4);
//				load_ptr=(tries<<4)+beg_pos_s[proc_vert]+(lane_id%16);
//				depth_s[(proc_vert<<4)+(lane_id%16)]=tex1Dfetch(tex_depth,load_ptr);
//			}
//			__syncthreads();
//			++tries;
//
//			//thread strid checking
//			for(int i = 0; i< 16; i++)
//				if(depth_curr == INFTY)//maybe unvisited
//					if(card_curr > 0){
//						if(card_curr < sml_shed){
//							ex_cat_sml_q[EX_OFF+ex_sml_ct]=tid;
//							ex_sml_ct++;
//						}else if(card_curr > lrg_shed){
//							ex_cat_lrg_q[EX_OFF+ex_lrg_ct]=tid;
//							ex_lrg_ct++;
//						}else{
//							ex_cat_mid_q[EX_OFF+ex_mid_ct]=tid;
//							ex_mid_ct++;
//						}
//					}
//			
//			//construct hub-cache
//			if(depth_curr == LEVEL){
//				hub_ptr	= tid & (HUB_BU_SZ - 1);
//				if(card_curr > hub_card[hub_ptr]){
//					hub_vert[hub_ptr] = tid;
//					hub_card[hub_ptr] = card_curr;
//				}
//			}
//			
//			card_curr	= card_next;
//			depth_curr	= depth_next;
//			tid			= tid_next;
//			tid_next	++;
//		}
//	
//	
//	}

#ifdef ENABLE_CHECKING
	if(	ex_sml_ct > bin_sz ||
		ex_mid_ct > bin_sz ||
		ex_lrg_ct > bin_sz)
		error_d = 1;
#endif
	
	ex_cat_sml_sz[TID_ST]= ex_sml_ct;
	ex_cat_mid_sz[TID_ST]= ex_mid_ct;
	ex_cat_lrg_sz[TID_ST]= ex_lrg_ct;
}

//+----------------------------
//|For xxx_inspect_clfy
//+----------------------------
template < typename vertex_t,typename index_t>
__global__ void reaper
(
	vertex_t *ex_cat_q,//sml, mid, lrg
	vertex_t *ex_q_d,//sml, mid, lrg
	index_t	 *ex_q_sz_d,
	index_t	 *ex_cat_off,
	const index_t bin_sz
)
{
	const index_t TID		= threadIdx.x+blockIdx.x*blockDim.x;
	const index_t SCAN_OFF	= ex_cat_off[TID];
	const index_t COUNT		= ex_q_sz_d[TID];
	const index_t BIN_OFF	= TID*bin_sz;
	
	for(index_t i=0;i<COUNT;i++)
		ex_q_d[SCAN_OFF+i]	= ex_cat_q[BIN_OFF+i];
}

template <typename vertex_t, typename index_t,
			typename depth_t>
__global__ void sort_bu_dist_inspect_clfy
(
	vertex_t *ex_cat_sml_q,//each thd obt ex_q 
	vertex_t *ex_cat_mid_q,//each thd obt ex_q 
	vertex_t *ex_cat_lrg_q,//each thd obt ex_q 
	index_t	 *ex_cat_sml_sz,
	index_t	 *ex_cat_mid_sz,
	index_t	 *ex_cat_lrg_sz,
	depth_t *depth_d,
	depth_t	 curr_level,
	index_t	 num_ver,
	const index_t sml_shed,
	const index_t lrg_shed,
	const index_t bin_sz
)
{
	const index_t TID_ST	= threadIdx.x +blockIdx.x*blockDim.x;
	const index_t NUM_VER	= num_ver;	
	const index_t GRNTY		= blockDim.x*gridDim.x;
	const index_t EX_OFF	= TID_ST*bin_sz;
	index_t tid_next		= TID_ST;
	index_t tid				= TID_ST;
	const depth_t 	LEVEL 	= curr_level;

	index_t card_curr, card_next;
	depth_t	depth_curr, depth_next;
	
	index_t ex_sml_ct	= 0;
	index_t ex_mid_ct	= 0;
	index_t ex_lrg_ct	= 0;
	index_t hub_ptr;
	
	while(tid < HUB_BU_SZ)
	{
		hub_vert[tid]	= V_INI_HUB;
		hub_card[tid]	= 0;

		tid	+= GRNTY;
	}
	tid	= TID_ST;

	if(tid < NUM_VER)
	{
		card_curr	= tex1Dfetch(tex_card, tid);
		depth_curr	= tex1Dfetch(tex_depth, tid);
		tid_next 	+= GRNTY;
	}
	
	while(tid<NUM_VER)
	{
		if(tid_next < NUM_VER)
		{
			card_next	= tex1Dfetch(tex_card, tid_next);
			depth_next	= tex1Dfetch(tex_depth, tid_next);
		}
		
		if(depth_curr == INFTY){
			//+---------------
			//|!!!HERE IS A FILTER OF CARD =0
			//+------------------------------
			if(card_curr<=0)
			{}else if(card_curr < sml_shed)
			{
				ex_cat_sml_q[EX_OFF+ex_sml_ct]=tid;
				ex_sml_ct++;
			}else if(card_curr > lrg_shed){
				ex_cat_lrg_q[EX_OFF+ex_lrg_ct]=tid;
				ex_lrg_ct++;
			}else{
				ex_cat_mid_q[EX_OFF+ex_mid_ct]=tid;
				ex_mid_ct++;
			}
		}

		//construct hub-cache
		if(depth_curr == LEVEL)
		{
			hub_ptr	= tid & (HUB_BU_SZ - 1);
			if(card_curr > hub_card[hub_ptr])
			{
				hub_vert[hub_ptr] = tid;
				hub_card[hub_ptr] = card_curr;
			}
		}

		depth_curr	= depth_next;
		card_curr	= card_next;
		tid			= tid_next;
		tid_next	+= GRNTY;
	}

#ifdef ENABLE_CHECKING
	if(	ex_sml_ct > bin_sz ||
		ex_mid_ct > bin_sz ||
		ex_lrg_ct > bin_sz)
		error_d = 1;
#endif
	
	ex_cat_sml_sz[TID_ST]= ex_sml_ct;
	ex_cat_mid_sz[TID_ST]= ex_mid_ct;
	ex_cat_lrg_sz[TID_ST]= ex_lrg_ct;
}


//+---------------------------------------------------
//|CLASSIFIED BASED STORAGE FOR EX_QUEUE
//+---------------------------------------------------
template< 	typename vertex_t, 
			typename index_t,
			typename depth_t>
void sort_inspect_clfy
(
	vertex_t *ex_cat_sml_q,//each thd obt ex_q 
	vertex_t *ex_cat_mid_q,//each thd obt ex_q 
	vertex_t *ex_cat_lrg_q,//each thd obt ex_q 
	vertex_t *ex_q_sml_d,
	vertex_t *ex_q_mid_d,
	vertex_t *ex_q_lrg_d,
	vertex_t *ex_cat_sml_sz,
	vertex_t *ex_cat_mid_sz,
	vertex_t *ex_cat_lrg_sz,
	vertex_t *ex_cat_sml_off,
	vertex_t *ex_cat_mid_off,
	vertex_t *ex_cat_lrg_off,
	depth_t	 *depth_d,
	depth_t	 curr_level,
	index_t	 *tr_edges_c_d,
	index_t	 *tr_edges_c_h,
	index_t  num_ver,//num ver in the graph
	cudaStream_t *stream,
	const index_t sml_shed,
	const index_t lrg_shed,
	const index_t bin_sz
)
{
	sort_dist_inspect_clfy
	<vertex_t, index_t, depth_t>
	<<<BLKS_NUM, THDS_NUM>>>
	(
		ex_cat_sml_q,//each thd obt ex_q 
		ex_cat_mid_q,//each thd obt ex_q 
		ex_cat_lrg_q,//each thd obt ex_q 
		tr_edges_c_d,
		ex_cat_sml_sz,
		ex_cat_mid_sz,
		ex_cat_lrg_sz,
		depth_d,
		curr_level,
		num_ver,
		sml_shed,
		lrg_shed,
		bin_sz
	);
	cudaThreadSynchronize();
	
	if(curr_level){
		cudaMemcpy(tr_edges_c_h, tr_edges_c_d, 
				sizeof(index_t)*BLKS_NUM, 
				cudaMemcpyDeviceToHost);
		
		//+------------------------------------
		//|SUM NUMBER OF VER TO BE EXPANDED
		//+--------------------------------------
		for(index_t i=1; i < BLKS_NUM; i++)
			tr_edges_c_h[0]	+= tr_edges_c_h[i];

		if(tr_edges_c_h[0] >= EDGES_C*0.3)
		{
			ENABLE_BTUP	= true;
#ifdef ENABLE_MONITORING
			std::cout<<"~~~~~~~TOP-DOWN-->>BOTTOM-UP~~~~~~~~~\n";	
#endif
			sort_switch_dist_inspect_clfy
			<vertex_t, index_t, depth_t>
			<<<BLKS_NUM, THDS_NUM>>>
			(
				ex_cat_sml_q,//each thd obt ex_q 
				ex_cat_mid_q,//each thd obt ex_q 
				ex_cat_lrg_q,//each thd obt ex_q 
				ex_cat_sml_sz,
				ex_cat_mid_sz,
				ex_cat_lrg_sz,
				depth_d,
				curr_level,
				num_ver,
				sml_shed,
				lrg_shed,
				bin_sz
			);
		}
	}
	
	//+------------------
	//|SML_Q
	//+------------------
	insp_scan
	<vertex_t, index_t>
	(
		ex_cat_sml_sz,
		ex_cat_sml_off,
		THDS_NUM*BLKS_NUM,
		BLKS_NUM>>1,
		THDS_NUM>>1,
		SML_Q,
		stream[0]
	);
	
	reaper
	<vertex_t, index_t>
	<<<BLKS_NUM, THDS_NUM, 0, stream[0]>>>
	(
		ex_cat_sml_q,
		ex_q_sml_d,
		ex_cat_sml_sz,
		ex_cat_sml_off,
		bin_sz
	);
	
	//+------------------
	//|MID_Q
	//+------------------
	insp_scan
	<vertex_t, index_t>
	(
		ex_cat_mid_sz,
		ex_cat_mid_off,
		THDS_NUM*BLKS_NUM,
		BLKS_NUM>>1,
		THDS_NUM>>1,
		MID_Q,
		stream[1]
	);
	reaper
	<vertex_t, index_t>
	<<<BLKS_NUM, THDS_NUM, 0, stream[1]>>>
	(
		ex_cat_mid_q,
		ex_q_mid_d,
		ex_cat_mid_sz,
		ex_cat_mid_off,
		bin_sz
	);
	
	//+------------------
	//|LRG_Q
	//+------------------
	insp_scan
	<vertex_t, index_t>
	(
		ex_cat_lrg_sz,
		ex_cat_lrg_off,
		THDS_NUM*BLKS_NUM,
		BLKS_NUM>>1,
		THDS_NUM>>1,
		LRG_Q,
		stream[2]
	);
	reaper
	<vertex_t, index_t>
	<<<BLKS_NUM, THDS_NUM, 0, stream[2]>>>
	(
		ex_cat_lrg_q,
		ex_q_lrg_d,
		ex_cat_lrg_sz,
		ex_cat_lrg_off,
		bin_sz
	);
	
	//+------------------------------------
	//|counting number of edges traversed
	//+------------------------------------
#ifdef ENABLE_MONITORING
	if(curr_level)
		std::cout<<"curr exp_sz:  "<<tr_edges_c_h[0]<<"vs"<<EDGES_C
				<<"="<<(tr_edges_c_h[0]*1.0/EDGES_C)
				<<"\n";
#endif

}

template< 	typename vertex_t, 
			typename index_t,
			typename depth_t>
void sort_bu_inspect_clfy
(
	vertex_t *ex_cat_sml_q,//each thd obt ex_q 
	vertex_t *ex_cat_mid_q,//each thd obt ex_q 
	vertex_t *ex_cat_lrg_q,//each thd obt ex_q 
	vertex_t *ex_q_sml_d,
	vertex_t *ex_q_mid_d,
	vertex_t *ex_q_lrg_d,
	vertex_t *ex_cat_sml_sz,
	vertex_t *ex_cat_mid_sz,
	vertex_t *ex_cat_lrg_sz,
	vertex_t *ex_cat_sml_off,
	vertex_t *ex_cat_mid_off,
	vertex_t *ex_cat_lrg_off,
	depth_t	 *depth_d,
	depth_t	 curr_level,
	index_t  num_ver,//num ver in the graph
	cudaStream_t *stream,
	const index_t sml_shed,
	const index_t lrg_shed,
	const index_t bin_sz
)
{
//	std::cout<<"Before catgorizer: "<<cudaDeviceSynchronize()<<"\n";
	sort_bu_dist_inspect_clfy
	<vertex_t, index_t, depth_t>
	<<<BLKS_NUM, THDS_NUM>>>
	(
		ex_cat_sml_q,//each thd obt ex_q 
		ex_cat_mid_q,//each thd obt ex_q 
		ex_cat_lrg_q,//each thd obt ex_q 
		ex_cat_sml_sz,
		ex_cat_mid_sz,
		ex_cat_lrg_sz,
		depth_d,
		curr_level,
		num_ver,
		sml_shed,
		lrg_shed,
		bin_sz
	);
	cudaThreadSynchronize();

	//+------------------
	//|SML_Q
	//+------------------
	insp_scan
	<vertex_t, index_t>
	(
		ex_cat_sml_sz,
		ex_cat_sml_off,
		THDS_NUM*BLKS_NUM,
		BLKS_NUM>>1,
		THDS_NUM>>1,
		SML_Q,
		stream[0]
	);
//	std::cout<<"After scan: "<<cudaDeviceSynchronize()<<"\n";
	
	reaper
	<vertex_t, index_t>
	<<<BLKS_NUM, THDS_NUM, 0, stream[0]>>>
	(
		ex_cat_sml_q,
		ex_q_sml_d,
		ex_cat_sml_sz,
		ex_cat_sml_off,
		bin_sz
	);
	
	//+------------------
	//|MID_Q
	//+------------------
	insp_scan
	<vertex_t, index_t>
	(
		ex_cat_mid_sz,
		ex_cat_mid_off,
		THDS_NUM*BLKS_NUM,
		BLKS_NUM>>1,
		THDS_NUM>>1,
		MID_Q,
		stream[1]
	);
	reaper
	<vertex_t, index_t>
	<<<BLKS_NUM, THDS_NUM, 0, stream[1]>>>
	(
		ex_cat_mid_q,
		ex_q_mid_d,
		ex_cat_mid_sz,
		ex_cat_mid_off,
		bin_sz
	);
	
	//+------------------
	//|LRG_Q
	//+------------------
	insp_scan
	<vertex_t, index_t>
	(
		ex_cat_lrg_sz,
		ex_cat_lrg_off,
		THDS_NUM*BLKS_NUM,
		BLKS_NUM>>1,
		THDS_NUM>>1,
		LRG_Q,
		stream[2]
	);
	reaper
	<vertex_t, index_t>
	<<<BLKS_NUM, THDS_NUM, 0, stream[2]>>>
	(
		ex_cat_lrg_q,
		ex_q_lrg_d,
		ex_cat_lrg_sz,
		ex_cat_lrg_off,
		bin_sz
	);
}
