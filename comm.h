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
#ifndef	COMM_HEADER
#define	COMM_HEADER

#include <stdio.h>
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", \
        cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define H_ERR( err ) \
  (HandleError( err, __FILE__, __LINE__ ))





#define MAX_Q_SIZE		(1024*1024)

//////////////////////////////////////////////////
//SCALE*THDS_NUMS*sizeof(int) should be 
//limited by the size of the shared memory
/////////////////////////////////////////////////
#define SCALE			(1024*4)	
//////////////////////////////////////////////////
#define	THDS_NUM			256	
#define	BLKS_NUM			256	

#define	V_NON_INC			-1
#define V_INI_HUB			-2

#define VALIDATE_TIMES		1
//--------------------------
typedef 	int 			index_t;
typedef		int				vertex_t;
typedef 	unsigned char depth_t;

typedef struct vertex
{
	index_t src_ver;
	index_t *conn_ver_list;
	index_t num_conn_ver;
	index_t depth;
}*v_st_t;

typedef struct	queue
{
	volatile index_t num_ver;
	index_t *front_queue;//only remember the vertex 
					//id other than the whole vertex
} queue_t;

enum in_q_t
{
	scan_based,	//Pack all the scan results together
	sort_based	//Static sort all adj-list vertices
};

enum inspect_t
{
	clsf_based,	//Classify expansion candidates
				//	into separate expansion queues:
				//	sml, mid and lrg queues
	
	norm_based	//Put all expansion candidates
				//	in one expansion queue
};

enum bfs_t
{
	top_down,	//Frontier to next level bfs	
	bottom_up	//Unknown vertices explore frontier
};

enum ex_q_t
{
	SML_Q,
	MID_Q,
	LRG_Q,
	NONE
};
//--------------------------------
#define	VIS				0x02
#define UNVIS			0x00
#define FRT				0x01
#define	SET_VIS(a)		((a)=0x02)

#define	SET_FRT(a)		((a)=0x01)

#define	IS_FRT(a)		((a)==0x01)
#define	IS_VIS(a)		((a)==0x02)
#define	IS_UNVIS(a)		((a)==0x00)

//----------------------------------
//GLOBAL VARIABLES
//---------------------------------
//--------------------------------
#define INFTY			(unsigned char)(0xffff)	
#define NUM_SRC		64
#endif

#ifndef EXTERN
#define EXTERN
index_t	in_q_sz;//cardinality of in_queue
__device__ index_t in_q_sz_d;
index_t	ex_q_sz;//cardinality of ex_queue
__device__ index_t ex_q_sz_d;
//-------------------
index_t	ex_sml_sz;//cardinality of ex_queue
index_t	ex_mid_sz;//cardinality of ex_queue
index_t	ex_lrg_sz;//cardinality of ex_queue
index_t	ex_cpu_q_sz;//cardinality of ex_queue
__device__ index_t ex_sml_sz_d;
__device__ index_t ex_mid_sz_d;
__device__ index_t ex_lrg_sz_d;

index_t ENABLE_CGU;
index_t ENABLE_BTUP;
index_t agg_tr_edges;//already traversed edges
index_t EDGES_C;//total edges in the graph

index_t	BIN_SZ	= 512;

__device__ index_t error_d;
index_t error_h;

//texture <vertex_t,	1, cudaReadModeElementType> tex_adj_list[4];
//size_t  tex_adj_off[4];

texture <index_t, 	1, cudaReadModeElementType> tex_card;
texture <index_t, 	1, cudaReadModeElementType> tex_strt;
texture <depth_t, 	1, cudaReadModeElementType> tex_depth;

texture <vertex_t,	1, cudaReadModeElementType> tex_sml_exq;
texture <vertex_t,	1, cudaReadModeElementType> tex_mid_exq;
texture <vertex_t,	1, cudaReadModeElementType> tex_lrg_exq;


__device__	index_t		hub_vert[1920];
__device__ 	depth_t		hub_stat[1920];
__device__	index_t		hub_card[1920];

#define HUB_SZ			1536	
#define HUB_BU_SZ		1920//should be 1.25 of HUB_SZ
								//since there is no status
								//array in __shared__ mem
#define HUB_CRITERIA	0	

#define	Q_CARD	3

#endif
