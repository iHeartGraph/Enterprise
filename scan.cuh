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
#define NUM_BANKS           32
#define LOG_NUM_BANKS       5

#define CONFLICT_FREE_OFFSET(n) \
    ((n) >>LOG_NUM_BANKS)

//USED FOR EXPANSION
template <typename data_t, typename index_t>
__global__ void __pre_scan(	
								data_t 	*scan_ind_d,
								data_t	*adj_card_d,
								data_t 	*scan_out_d,
								data_t 	*blk_sum,
								index_t num_dat,
								const index_t THD_NUM
		   				   )
{
	const data_t tile_sz	= THD_NUM<<1;
	const index_t lane		= threadIdx.x<<1;
	index_t tid				= threadIdx.x+blockIdx.x*blockDim.x;
	index_t proc_tile		= blockIdx.x;

	index_t offset			= 1;
	index_t tid_strip;
	index_t idct_a, idct_b;
	const index_t padding	= CONFLICT_FREE_OFFSET(tile_sz - 1);
	const index_t NUM_THS	= gridDim.x*blockDim.x;

	//conflict free
	const index_t off_a		= CONFLICT_FREE_OFFSET(lane);
	const index_t off_b		= CONFLICT_FREE_OFFSET(lane+1);
	index_t num_tiles		= num_dat/tile_sz;
	if(num_dat%tile_sz)
		num_tiles++;

	//prefetching danger
	extern __shared__ data_t s_mem[];
	//__shared__ data_t s_mem[280];
	
	while(proc_tile < num_tiles)
	{
		tid_strip = tid<<1;
		if(tid_strip < num_dat)
		{
			idct_a	= scan_ind_d[tid_strip];
			s_mem[lane+off_a]= adj_card_d[idct_a];
		}else{
			s_mem[lane+off_a]= 0;
		}

		if((tid_strip + 1) < num_dat)
		{
			idct_b	= scan_ind_d[tid_strip + 1];
			s_mem[lane + 1 + off_b]	= adj_card_d[idct_b];
		}else{
			s_mem[lane + 1 + off_b] = 0;
		}
		__syncthreads();

		//up sweep
		for(index_t j = THD_NUM;j > 0;j>>=1)
		{   
			if(threadIdx.x < j)
			{
				index_t ai	= offset*lane +offset - 1;
				index_t bi	= ai + offset;
				ai		   += CONFLICT_FREE_OFFSET(ai);
				bi		   += CONFLICT_FREE_OFFSET(bi);
				s_mem[bi]  += s_mem[ai];
			}
			offset <<=1;
			__syncthreads();
		}   
		__syncthreads();

		//write the block sum
		if(blockDim.x > 1)//save sml scan job
		{
			if(threadIdx.x == 0)
			{
				blk_sum[tid/THD_NUM]	= 
							s_mem[tile_sz -1 + padding];
				s_mem[tile_sz-1 + padding]		= 0;
			}
			__syncthreads();
		}

		//down sweep
		for(index_t j=1; j < tile_sz; j <<=1)
		{
			offset	>>=	1;
			if(threadIdx.x < j)
			{
				index_t ai	= lane*offset + offset - 1;
				index_t bi	= ai + offset;
				ai		   += CONFLICT_FREE_OFFSET(ai);
				bi		   += CONFLICT_FREE_OFFSET(bi);
				index_t t 	= s_mem[ai];
				s_mem[ai]	= s_mem[bi];
				s_mem[bi]  += t;
			}
			__syncthreads();
		}
		__syncthreads();

		//store data
		if(tid_strip < num_dat)
		{
			scan_out_d[tid_strip]	= s_mem[lane+off_a]; 
		}
		
		if((tid_strip + 1) < num_dat)
		{
			scan_out_d[tid_strip+1] 	= s_mem[lane+1+off_b]; 
		}

		tid			+= NUM_THS;
		proc_tile	+= gridDim.x;
	}
}

//USED FOR lrg_scan
template <typename data_t, typename index_t>
__global__ void __spine_scan(
								data_t 	*blk_sum, 
								data_t 	*grd_sum,
								index_t num_dat,
								const index_t THD_NUM
		   				   )
{
	extern __shared__ data_t s_mem[];

	const data_t tile_sz	= THD_NUM<<1;
	const index_t lane		= threadIdx.x<<1;
	const index_t GRNTY= blockDim.x*gridDim.x;

	const index_t padding	= CONFLICT_FREE_OFFSET(tile_sz - 1);
	const index_t off_a		= CONFLICT_FREE_OFFSET(lane);
	const index_t off_b		= CONFLICT_FREE_OFFSET(lane+1);
	
	index_t tid				= threadIdx.x+blockIdx.x*blockDim.x;
	index_t offset			= 1;
	index_t	tid_strip		= tid<<1;
	index_t proc_tile		= blockIdx.x;
		
	index_t num_tiles		= num_dat/tile_sz;
	if(num_dat%tile_sz)
		num_tiles++;

	while(proc_tile < num_tiles)
	{	
		tid_strip		= tid<<1;
		
		if(tid_strip < num_dat)
			s_mem[lane+off_a] = blk_sum[tid_strip];
		else
			s_mem[lane+off_a]	= 0;

		if(tid_strip < num_dat)
			s_mem[lane+1+off_b]	= blk_sum[tid_strip + 1];
		else
			s_mem[lane+1+off_b] = 0;
	
		__syncthreads();

		//up sweep
		for(index_t j = THD_NUM;j > 0;j>>=1)
		{   
			if(threadIdx.x < j)
			{
				index_t ai	= offset*lane +offset - 1;
				index_t bi	= ai + offset;

				ai		   += CONFLICT_FREE_OFFSET(ai);
				bi		   += CONFLICT_FREE_OFFSET(bi);
				s_mem[bi] += s_mem[ai];
			}
			offset <<=1;
			__syncthreads();
		}   
		__syncthreads();

		//write the block sum
		if(threadIdx.x == 0)
		{
			grd_sum[tid/THD_NUM]= 
						s_mem[tile_sz-1+padding];
			s_mem[tile_sz-1+padding]= 0;
		}
		__syncthreads();

		//down sweep
		for(index_t j=1; j < tile_sz; j <<=1)
		{
			offset	>>=	1;
			if(threadIdx.x < j)
			{
				index_t ai	= lane*offset + offset - 1;
				index_t bi	= ai + offset;
				ai		   += CONFLICT_FREE_OFFSET(ai);
				bi		   += CONFLICT_FREE_OFFSET(bi);
				index_t t 	= s_mem[ai];
				s_mem[ai]	= s_mem[bi];
				s_mem[bi]  += t;
			}
			__syncthreads();
		}
		__syncthreads();
		if(tid_strip < num_dat)
			blk_sum[tid_strip]		= s_mem[lane+off_a];
		if(tid_strip + 1< num_dat)
			blk_sum[tid_strip+1]	= s_mem[lane+1+off_b];
		
		tid			+= GRNTY;
		proc_tile	+= gridDim.x;
	}
}

/////////////////////////////////////////////////////
//post_scan for large problem
//////////////////////////////////////////////////////
template<typename data_t, typename index_t>
__global__ void __post_scan(
								data_t 	*scan_out_d,
								data_t 	*blk_sum,
								data_t 	*grd_sum,
								index_t	num_dat,
								index_t num_grd,
								const index_t THD_NUM
							)
{
	extern __shared__ data_t s_mem[];
	
	const data_t tile_sz	= THD_NUM<<1;
	const index_t lane		= threadIdx.x<<1;
	index_t tid				= threadIdx.x+blockIdx.x*blockDim.x;
	index_t offset			= 1;
	
	const index_t padding	= CONFLICT_FREE_OFFSET(tile_sz - 1);
	
	const index_t off_a		= CONFLICT_FREE_OFFSET(lane);
	const index_t off_b		= CONFLICT_FREE_OFFSET(lane+1);
	/////////////////////////////////////////////////////////
	//GRID SCAN
	/////////////////////////////////////////////////////////
	if(lane < num_grd)
		s_mem[lane+off_a]=grd_sum[lane];
	else
		s_mem[lane+off_a]=0;

	if(lane+1 < num_grd)
		s_mem[lane+1+off_b]=grd_sum[lane+1];
	else
		s_mem[lane+1+off_b]=0;
	
	__syncthreads();
	
	//up sweep
	for(index_t j = THD_NUM;j > 0;j>>=1)
	{   
		if(threadIdx.x < j)
		{
			index_t ai	= offset*lane +offset - 1;
			index_t bi	= ai + offset;

			ai		   += CONFLICT_FREE_OFFSET(ai);
			bi		   += CONFLICT_FREE_OFFSET(bi);
			s_mem[bi] += s_mem[ai];
		}
		offset <<=1;
		__syncthreads();
	}   
	__syncthreads();

	//write the block sum
	if(threadIdx.x == 0)
	{
		if(!blockIdx.x)
			in_q_sz_d	= s_mem[tile_sz-1+padding];
		
		s_mem[tile_sz-1+padding]= 0;
	}
	__syncthreads();

	//down sweep
	for(index_t j=1; j < tile_sz; j <<=1)
	{
		offset	>>=	1;
		if(threadIdx.x < j)
		{
			index_t ai	= lane*offset + offset - 1;
			index_t bi	= ai + offset;
			ai		   += CONFLICT_FREE_OFFSET(ai);
			bi		   += CONFLICT_FREE_OFFSET(bi);
			index_t t 	= s_mem[ai];
			s_mem[ai]	= s_mem[bi];
			s_mem[bi]  += t;
		}
		__syncthreads();
	}
	__syncthreads();

	const index_t P_DATA	= gridDim.x*(blockDim.x<<1);
	tid 		<<= 1;
	index_t blk_idx	= blockIdx.x;
	
	index_t grd_idx_orig= 0;
	index_t grd_idx		= grd_idx_orig;
			//should add the offset but we eliminate 
			//since CONFLICT_FREE_OFFSET(0) = 0;
	
	while (tid<num_dat)
	{
		scan_out_d[tid]		= 	scan_out_d[tid]+ 
								blk_sum[blk_idx]+ 
								s_mem[grd_idx];	
		if(tid+1<num_dat)
			scan_out_d[tid+1]=scan_out_d[tid+1]+ 
								blk_sum[blk_idx]+ 
								s_mem[grd_idx];

		tid		+= P_DATA;

		blk_idx	+= gridDim.x;
		if(!((blk_idx -blockIdx.x)%(gridDim.x<<1)))
		{
			grd_idx_orig++;
			grd_idx	= grd_idx_orig + 
				CONFLICT_FREE_OFFSET(grd_idx_orig);
		}
	}
}


//mid problem
template<typename data_t, typename index_t>
__global__ void __post_scan(
								data_t 	*scan_out_d,
								data_t 	*blk_sum,
								index_t	num_dat,
								index_t num_blk,
								const index_t THD_NUM
							)
{
	extern __shared__ data_t s_mem[];
	
	const data_t tile_sz	= THD_NUM<<1;
	const index_t lane		= threadIdx.x<<1;
	index_t tid				= threadIdx.x+blockIdx.x*blockDim.x;
	index_t offset			= 1;
	const index_t padding	= CONFLICT_FREE_OFFSET(tile_sz - 1);
	
	const index_t off_a		= CONFLICT_FREE_OFFSET(lane);
	const index_t off_b		= CONFLICT_FREE_OFFSET(lane+1);
	/////////////////////////////////////////////////////////
	//GRID SCAN
	/////////////////////////////////////////////////////////
	if(lane < num_blk)
	{
		s_mem[lane+off_a]=blk_sum[lane];
	}else{
		s_mem[lane+off_a]=0;
	}

	if(lane+1 < num_blk)
	{
		s_mem[lane+1+off_b]=blk_sum[lane+1];
	}else{
		s_mem[lane+1+off_b]=0;
	}
	__syncthreads();
	
	//up sweep
	for(index_t j = THD_NUM;j > 0;j>>=1)
	{   
		if(threadIdx.x < j)
		{
			index_t ai	= offset*lane +offset - 1;
			index_t bi	= ai + offset;

			ai		   += CONFLICT_FREE_OFFSET(ai);
			bi		   += CONFLICT_FREE_OFFSET(bi);
			s_mem[bi] += s_mem[ai];
		}
		offset <<=1;
		__syncthreads();
	}   
	__syncthreads();

	//write the block sum
	if(!threadIdx.x)
	{
		if(!blockIdx.x)
			in_q_sz_d	= s_mem[tile_sz-1+padding]; 
		
		s_mem[tile_sz+padding-1]	= 0;
	}
	__syncthreads();

	//down sweep
	for(index_t j=1; j < tile_sz; j <<=1)
	{
		offset	>>=	1;
		if(threadIdx.x < j)
		{
			index_t ai	= lane*offset + offset - 1;
			index_t bi	= ai + offset;
			ai		   += CONFLICT_FREE_OFFSET(ai);
			bi		   += CONFLICT_FREE_OFFSET(bi);
			index_t t 	= s_mem[ai];
			s_mem[ai]	= s_mem[bi];
			s_mem[bi]  += t;
		}
		__syncthreads();
	}
	__syncthreads();

	tid 				<<= 1;
	index_t blk_idx_orig= blockIdx.x;
	index_t blk_idx		= blk_idx_orig + 
				CONFLICT_FREE_OFFSET(blk_idx_orig);
	const index_t P_DATA= blockDim.x*(gridDim.x<<1);
	
	while (tid<num_dat)
	{
		scan_out_d[tid]		+=s_mem[blk_idx];	
		
		if(tid+1<num_dat)
			scan_out_d[tid+1]+= s_mem[blk_idx];

		tid			+= P_DATA;
		blk_idx_orig+= gridDim.x;
		blk_idx		= blk_idx_orig +
					CONFLICT_FREE_OFFSET(blk_idx_orig);
//-------------------------------------------------
//KEEP THIS AS WELL
//this time we substract too many CONFLICT_FREE_OFF 
//		blk_idx	-= CONFLICT_FREE_OFFSET(blk_idx);
//		blk_idx	+= gridDim.x;
//		blk_idx	+= CONFLICT_FREE_OFFSET(blk_idx);
//
//-------------------------------------------------
//KEEP IT HERE:
//	if this way of counting blk_idx, we count 
//the previous padding twice!!!!
//		blk_idx	= gridDim.x;
//		blk_idx	+= CONFLICT_FREE_OFFSET(blk_idx);
//---------------------------------------------------
	}
}


//USED FOR INSPECTION
template <typename data_t, typename index_t>
__global__ void __insp_pre_scan(	
								data_t 	*scan_in_d, 
								data_t 	*scan_out_d,
								data_t 	*blk_sum,
								index_t num_dat,
								const index_t THD_NUM
		   				   )
{
	const data_t tile_sz	= THD_NUM<<1;
	const index_t lane		= threadIdx.x<<1;
	index_t tid				= threadIdx.x+blockIdx.x*blockDim.x;
	index_t offset			= 1;
	index_t	tid_strip		= tid<<1;
	const index_t padding	= CONFLICT_FREE_OFFSET(tile_sz - 1 );
	
	//conflict free
	const index_t off_a		= CONFLICT_FREE_OFFSET(lane);
	const index_t off_b		= CONFLICT_FREE_OFFSET(lane+1);
	const index_t GRNTY = blockDim.x*gridDim.x;


	//prefetching danger
	extern __shared__ data_t s_mem[];
	
	
	while(tid_strip < num_dat)
	{
		s_mem[lane+off_a] 		= scan_in_d[tid_strip];
		s_mem[lane + 1 + off_b]	= scan_in_d[tid_strip + 1];
		__syncthreads();
		
		//up sweep
		for(index_t j = THD_NUM;j > 0;j>>=1)
		{   
			if(threadIdx.x < j)
			{
				index_t ai	= offset*lane +offset - 1;
				index_t bi	= ai + offset;
				ai		   += CONFLICT_FREE_OFFSET(ai);
				bi		   += CONFLICT_FREE_OFFSET(bi);
				s_mem[bi]  += s_mem[ai];
			}
			offset <<=1;
			__syncthreads();
		}   
		__syncthreads();

		//write the block sum
		if(threadIdx.x == 0)
		{
			blk_sum[tid/THD_NUM]	= 
							s_mem[tile_sz -1 + padding];
			s_mem[tile_sz-1 + padding]		= 0;
		}
		__syncthreads();

		//down sweep
		for(index_t j=1; j < tile_sz; j <<=1)
		{
			offset	>>=	1;
			if(threadIdx.x < j)
			{
				index_t ai	= lane*offset + offset - 1;
				index_t bi	= ai + offset;
				ai		   += CONFLICT_FREE_OFFSET(ai);
				bi		   += CONFLICT_FREE_OFFSET(bi);
				index_t t 	= s_mem[ai];
				s_mem[ai]	= s_mem[bi];
				s_mem[bi]  += t;
			}
			__syncthreads();
		}
		__syncthreads();
		scan_out_d[tid_strip]	= s_mem[lane + off_a];
		scan_out_d[tid_strip+1]	= s_mem[lane + 1 + off_b];
		
		tid += GRNTY;
		tid_strip = tid<<1;
	}
}

////////////////////////////////////////////////////////
//post_scan for inspection
////////////////////////////////////////////////////////
template<typename data_t, typename index_t>
__global__ void __insp_post_scan(
								data_t 	*scan_out_d,
								data_t 	*blk_sum,
								index_t	num_dat,
								index_t num_blk,
								const index_t THD_NUM,
								const ex_q_t q_t
							)
{
	extern __shared__ data_t s_mem[];
	
	const data_t tile_sz	= THD_NUM<<1;
	const index_t lane		= threadIdx.x<<1;
	index_t tid				= threadIdx.x+blockIdx.x*blockDim.x;
	index_t offset			= 1;
	
	const index_t padding	= CONFLICT_FREE_OFFSET(tile_sz - 1);
	
	const index_t off_a		= CONFLICT_FREE_OFFSET(lane);
	const index_t off_b		= CONFLICT_FREE_OFFSET(lane+1);
	/////////////////////////////////////////////////////////
	//GRID SCAN
	/////////////////////////////////////////////////////////
	if(lane < num_blk)
	{
		s_mem[lane+off_a]=blk_sum[lane];
	}else{
		s_mem[lane+off_a]=0;
	}

	if(lane+1 < num_blk)
	{
		s_mem[lane+1+off_b]=blk_sum[lane+1];
	}else{
		s_mem[lane+1+off_b]=0;
	}
	__syncthreads();
	
	//up sweep
	for(index_t j = THD_NUM;j > 0;j>>=1)
	{   
		if(threadIdx.x < j)
		{
			index_t ai	= offset*lane +offset - 1;
			index_t bi	= ai + offset;

			ai		   += CONFLICT_FREE_OFFSET(ai);
			bi		   += CONFLICT_FREE_OFFSET(bi);
			s_mem[bi] += s_mem[ai];
		}
		offset <<=1;
		__syncthreads();
	}   
	__syncthreads();

	//write the block sum
	if(!threadIdx.x)
	{
	
	if(!blockIdx.x)
	{
		switch(q_t)
		{
			case SML_Q:
				ex_sml_sz_d	= s_mem[tile_sz-1+padding];
				break;
			case MID_Q:
				ex_mid_sz_d	= s_mem[tile_sz-1+padding];
				break;
			case LRG_Q:
				ex_lrg_sz_d	= s_mem[tile_sz-1+padding];
				break;
			default:
				break;
		}
	}
		s_mem[tile_sz-1+padding]= 0;
	}
	__syncthreads();

	//down sweep
	for(index_t j=1; j < tile_sz; j <<=1)
	{
		offset	>>=	1;
		if(threadIdx.x < j)
		{
			index_t ai	= lane*offset + offset - 1;
			index_t bi	= ai + offset;
			ai		   += CONFLICT_FREE_OFFSET(ai);
			bi		   += CONFLICT_FREE_OFFSET(bi);
			index_t t 	= s_mem[ai];
			s_mem[ai]	= s_mem[bi];
			s_mem[bi]  += t;
		}
		__syncthreads();
	}
	__syncthreads();

	tid 			<<= 1;
	index_t blk_idx_orig= blockIdx.x;
	index_t blk_idx		= blk_idx_orig + 
				CONFLICT_FREE_OFFSET(blk_idx_orig);
	const index_t P_DATA= gridDim.x*blockDim.x*2;

	while (tid<num_dat)
	{
		scan_out_d[tid]		= 	scan_out_d[tid] + 
								s_mem[blk_idx];	
		scan_out_d[tid + 1]	= 	scan_out_d[tid + 1] + 
								s_mem[blk_idx];
		tid			+= P_DATA;
		blk_idx_orig+= gridDim.x;
		blk_idx		= blk_idx_orig +
					CONFLICT_FREE_OFFSET(blk_idx_orig);
	}
}

template<typename data_t, typename index_t>
__host__ void lrg_scan(
						data_t 			*scan_ind_d,
						data_t			*adj_card_d,
						//TODO requires scan_in_d to be 
						//		exact times of 
						//		THD_NUM*2
						data_t 			*scan_out_d,
						const index_t	BLK_NUM,
						const index_t 	THD_NUM,
						cudaStream_t 	&stream
		   		   		)
{
	const index_t num_dat = ex_q_sz;
	data_t *blk_sum;
	data_t *grd_sum;
	index_t num_grd;
	
	const size_t sz = sizeof(data_t);
	const index_t padding	= 
					CONFLICT_FREE_OFFSET((THD_NUM<<1) -1);
	index_t num_blk	= num_dat/(THD_NUM<<1);
	if (num_dat % (THD_NUM<<1))
		num_blk	++;
	
	num_grd = num_blk/(THD_NUM<<1);
	if(num_blk%(THD_NUM<<1))
		num_grd ++;

	cudaMalloc((void **)&blk_sum,
			sizeof(data_t)*num_blk);

	cudaMalloc((void **)&grd_sum,
			sizeof(data_t)*num_grd);
if(num_grd>(THD_NUM<<1))
	std::cout<<"Out of Range\n";
	__pre_scan<data_t, index_t>
	<<<BLK_NUM, THD_NUM, (padding+(THD_NUM<<1))*sz, stream>>>
	(
		scan_ind_d,
		adj_card_d,
		scan_out_d,
		blk_sum,
		num_dat,
		THD_NUM
	);
	cudaThreadSynchronize();
	__spine_scan<data_t, index_t>
	<<<BLK_NUM, THD_NUM, (padding+(THD_NUM<<1))*sz, stream>>>
	(
		blk_sum,
		grd_sum,
		num_blk,
		THD_NUM
	);
	cudaThreadSynchronize();
	__post_scan<data_t, index_t>
	<<<BLK_NUM, THD_NUM, (padding+(THD_NUM<<1))*sz, stream>>>
	(
		scan_out_d,
		blk_sum,
		grd_sum,
		num_dat,
		num_grd,
		THD_NUM
	);
	cudaThreadSynchronize();
}

template<typename data_t, typename index_t>
__host__ void mid_scan(
						data_t 			*scan_ind_d,
						data_t			*adj_card_d,
						data_t 			*scan_out_d,
						const index_t	BLK_NUM,
						const index_t 	THD_NUM,
						cudaStream_t 	&stream
		   		   		)
{
	const index_t num_dat = ex_q_sz;
	data_t *blk_sum;
	const size_t sz = sizeof(data_t);
	const index_t padding	= 
					CONFLICT_FREE_OFFSET((THD_NUM<<1) -1);
	index_t num_blk	= num_dat/(THD_NUM<<1);
	if(num_blk % (THD_NUM<<1))
		num_blk ++;
//	std::cout<<"mid problem, num dat: "<<num_dat<<"\n";
	cudaMalloc((void **)&blk_sum,
			sizeof(data_t)*num_blk);
	
	__pre_scan<data_t, index_t>
	<<<BLK_NUM, THD_NUM, (padding+(THD_NUM<<1))*sz, stream>>>
	(
		scan_ind_d,
		adj_card_d,
		scan_out_d,
		blk_sum,
		num_dat,
		THD_NUM
	);	
	
	cudaThreadSynchronize();
	__post_scan<data_t, index_t>
	<<<BLK_NUM, THD_NUM, (padding+(THD_NUM<<1))*sz, stream>>>
	(
		scan_out_d,
		blk_sum,
		num_dat,
		num_blk,
		THD_NUM
	);	
}

template<typename data_t, typename index_t>
__global__ void sml_scan(
						data_t 			*scan_ind_d,
						data_t 			*adj_card_d,
						data_t 			*scan_out_d,
						const index_t 	THD_NUM
		   		   		)
{
	
	const 	index_t	num_dat	= ex_q_sz_d;
	const data_t tile_sz	= THD_NUM<<1;
	const index_t lane		= threadIdx.x<<1;
	index_t offset			= 1;
	index_t idct_a, idct_b;
	const index_t padding	= 
				CONFLICT_FREE_OFFSET((THD_NUM<<1)-1);
	
	//conflict free
	const index_t off_a		= CONFLICT_FREE_OFFSET(lane);
	const index_t off_b		= CONFLICT_FREE_OFFSET(lane+1);

	//prefetching danger
	extern __shared__ data_t s_mem[];
	
	if(lane < num_dat)
	{
		idct_a	= scan_ind_d[lane];
		s_mem[lane+off_a]	= adj_card_d[idct_a]; 
	}else{
		s_mem[lane+off_a]	= 0;	
	}

	if((lane + 1) < num_dat)
	{
		idct_b	= scan_ind_d[lane + 1];
		s_mem[lane+1+off_b]	= adj_card_d[idct_b]; 
	}else{
		s_mem[lane+1+off_b]	= 0;	
	}
	__syncthreads();
	
	//up sweep
	for(index_t j = THD_NUM;j > 0;j>>=1)
	{   
		if(threadIdx.x < j)
		{
			index_t ai	= offset*lane +offset - 1;
			index_t bi	= ai + offset;
			ai		   += CONFLICT_FREE_OFFSET(ai);
			bi		   += CONFLICT_FREE_OFFSET(bi);
			s_mem[bi]  += s_mem[ai];
		}
		offset <<=1;
		__syncthreads();
	}   
	__syncthreads();

	if(!threadIdx.x)
	{
		in_q_sz_d	= s_mem[(THD_NUM<<1)+ padding -1];
		s_mem[(THD_NUM<<1) + padding -1]	= 0;
	}
	__syncthreads();
	
	//down sweep
	for(index_t j=1; j < tile_sz; j <<=1)
	{
		offset	>>=	1;
		if(threadIdx.x < j)
		{
			index_t ai	= lane*offset + offset - 1;
			index_t bi	= ai + offset;
			ai		   += CONFLICT_FREE_OFFSET(ai);
			bi		   += CONFLICT_FREE_OFFSET(bi);
			index_t t 	= s_mem[ai];
			s_mem[ai]	= s_mem[bi];
			s_mem[bi]  += t;
		}
		__syncthreads();
	}
	__syncthreads();
	
	if(lane < num_dat)
		scan_out_d[lane]	= s_mem[lane + off_a];
	
	if((lane+1) < num_dat)
		scan_out_d[lane +1]	= s_mem[lane + 1 + off_b];
}

//-----------------------------------------
//For inspection, 
//exact threads number of data to scan
//---------------------------------------
template<typename data_t, typename index_t>
__host__ void insp_scan(
						data_t 			*scan_in_d,
						//TODO requires scan_in_d to be 
						//		exact times of 
						//		THD_NUM*2
						data_t 			*scan_out_d,
						const index_t	num_dat,
						const index_t	BLK_NUM,
						const index_t	THD_NUM,
						const ex_q_t	q_t,
						cudaStream_t 	&stream
		   		   		)
{
	data_t *blk_sum;
	const size_t sz = sizeof(data_t);
	const index_t num_blk	= THD_NUM<<1;
	const index_t padding	= 
				CONFLICT_FREE_OFFSET((THD_NUM<<1)-1);
//	std::cout<<"padding = "<<padding<<"\n";
	cudaMalloc((void **)&blk_sum,
			sz*num_blk);
	
	__insp_pre_scan<data_t, index_t>
	<<<BLK_NUM, THD_NUM, (padding+(THD_NUM<<1))*sz, stream>>>
	(
		scan_in_d,
		scan_out_d,
		blk_sum,
		num_dat,
		THD_NUM
	);	
	
	cudaThreadSynchronize();
	__insp_post_scan<data_t, index_t>
	<<<BLK_NUM, THD_NUM, (padding+(THD_NUM<<1))*sz, stream>>>
	(
		scan_out_d,
		blk_sum,
		num_dat,
		num_blk,
		THD_NUM,
		q_t
	);	
}
