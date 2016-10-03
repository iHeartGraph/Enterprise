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
 * Hang Liu and H. Howie Huang. 2015. Enterprise: breadth-first graph traversal on GPUs. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '15). ACM, New York, NY, USA, , Article 68 , 12 pages. DOI: http://dx.doi.org/10.1145/2807591.2807594
 */
#include <iostream>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <assert.h>

#define INFTY int(1<<30)
using namespace std;

typedef long int vertex_t;
typedef long int index_t;

inline off_t fsize(const char *filename) {
    struct stat st; 
    if (stat(filename, &st) == 0)
        return st.st_size;
    return -1; 
}

		
int main(int argc, char** argv){
	int fd,i;
	char* ss_head;
	char* ss;
	
	std::cout<<"Input: ./exe tuple_file(text) "
			<<"reverse_the_edge(1 reverse, 0 not reverse) lines_to_skip\n";
	if(argc<4){printf("Wrong input\n");exit(-1);}
	
	size_t file_size = fsize(argv[1]);
	int is_reverse=(atol(argv[2])==1);	
	long skip_head=atol(argv[3]);
	

	fd=open(argv[1],O_CREAT|O_RDWR,00666 );
	if(fd == -1)
	{
		printf("%s open error\n", argv[1]);
		perror("open");
		exit(-1);
	}

	ss_head = (char*)mmap(NULL,file_size,PROT_READ|PROT_WRITE,MAP_PRIVATE,fd,0);
	assert(ss_head != MAP_FAILED);
	size_t head_offset=0;
	int skip_count = 0;
	while(true)
	{
		if(skip_count == skip_head) break;
		if(head_offset == file_size &&
				skip_count < skip_head)
		{
			std::cout<<"Eorr: skip more lines than the file has\n\n\n";
			exit(-1);
		}

		head_offset++;
		if(ss_head[head_offset]=='\n')
		{
			skip_count++;
			if(skip_count == skip_head) break;
		}
	}

	ss = &ss_head[head_offset];
	file_size -= head_offset;

	size_t curr=0;
	size_t next=0;

	//step 1. vert_count,edge_count,
	size_t edge_count=0;
	size_t vert_count;
	vertex_t v_max = 0;
	vertex_t v_min = INFTY;//as infinity
	vertex_t a;
	while(next<file_size){
		char* sss=ss+curr;
		a = atol(sss);
		if(v_max<a){
			v_max = a;
		}
		if(v_min>a){
			v_min = a;
		}

		while((ss[next]!=' ')&&(ss[next]!='\n')&&(ss[next]!='\t')){
			next++;
		}
		while((ss[next]==' ')||(ss[next]=='\n')||(ss[next]=='\t')){
			next++;
		}
		curr = next;
		
		//one vertex is counted once
		edge_count++;
	}
	
	const index_t line_count=edge_count>>1;
	if(!is_reverse) edge_count >>=1;
	
	vert_count = v_max - v_min + 1;
	assert(v_min<INFTY);
	cout<<"edge count: "<<edge_count<<endl;
	cout<<"max vertex id: "<<v_max<<endl;
	cout<<"min vertex id: "<<v_min<<endl;

	cout<<"edge count: "<<edge_count<<endl;
	cout<<"vert count: "<<vert_count<<endl;
	
	//step 2. each file size
	char filename[256];
	sprintf(filename,"%s_csr.bin",argv[1]);
	int fd4 = open(filename,O_CREAT|O_RDWR,00666 );
	ftruncate(fd4, edge_count*sizeof(vertex_t));
	vertex_t* adj = (vertex_t*)mmap(NULL,edge_count*sizeof(vertex_t),PROT_READ|PROT_WRITE,MAP_SHARED,fd4,0);
	assert(adj != MAP_FAILED);

	//added by Hang to generate a weight file
	sprintf(filename,"%s_weight.bin",argv[1]);
	int fd6 = open(filename,O_CREAT|O_RDWR,00666 );
	ftruncate(fd6, edge_count*sizeof(vertex_t));
	index_t* weight= (vertex_t*)mmap(NULL,edge_count*sizeof(vertex_t),PROT_READ|PROT_WRITE,MAP_SHARED,fd6,0);
	assert(weight != MAP_FAILED);
	//-End 

	sprintf(filename,"%s_head.bin",argv[1]);
	int fd5 = open(filename,O_CREAT|O_RDWR,00666 );
	ftruncate(fd5, edge_count*sizeof(vertex_t));
	vertex_t* head = (vertex_t*)mmap(NULL,edge_count*sizeof(vertex_t),PROT_READ|PROT_WRITE,MAP_SHARED,fd5,0);
	assert(head != MAP_FAILED);

	sprintf(filename,"%s_deg.bin",argv[1]);
	int fd2 = open(filename,O_CREAT|O_RDWR,00666 );
	ftruncate(fd2, vert_count*sizeof(index_t));
	index_t* degree = (index_t*)mmap(NULL,vert_count*sizeof(index_t),PROT_READ|PROT_WRITE,MAP_SHARED,fd2,0);
	assert(degree != MAP_FAILED);
	
	sprintf(filename,"%s_beg_pos.bin",argv[1]);
	int fd3 = open(filename,O_CREAT|O_RDWR,00666 );
	ftruncate(fd3, (vert_count+1)*sizeof(index_t));
	index_t* begin  = (index_t*)mmap(NULL,(vert_count+1)*sizeof(index_t),PROT_READ|PROT_WRITE,MAP_SHARED,fd3,0);
	assert(begin != MAP_FAILED);
	
	//step 3. write degree
	for(int i=0; i<vert_count;i++){
		degree[i]=0;
	}

	vertex_t index, dest;
	size_t offset =0;
	curr=0;
	next=0;

	printf("Getting degree...\n");
	while(offset<line_count){
		char* sss=ss+curr;
		index = atol(sss)-v_min;
		while((ss[next]!=' ')&&(ss[next]!='\n')&&(ss[next]!='\t')){
			next++;
		}
		while((ss[next]==' ')||(ss[next]=='\n')||(ss[next]=='\t')){
			next++;
		}
		curr = next;

		char* sss1=ss+curr;
		dest=atol(sss1)-v_min;

		while((ss[next]!=' ')&&(ss[next]!='\n')&&(ss[next]!='\t')){
			next++;
		}
		while((ss[next]==' ')||(ss[next]=='\n')||(ss[next]=='\t')){
			next++;
		}
		curr = next;
		degree[index]++;
		if(is_reverse) degree[dest]++;
//		cout<<index<<" "<<degree[index]<<endl;

		offset++;
	}
//	exit(-1);
	begin[0]=0;
	begin[vert_count]=edge_count;

	printf("Calculate beg_pos ...\n");
	for(size_t i=1; i<vert_count; i++){
		begin[i] = begin[i-1] + degree[i-1];
//		cout<<begin[i]<<" "<<degree[i]<<endl;
		degree [i-1] = 0;
	}
	degree[vert_count-1] = 0;
	//step 4: write adjacent list 
	vertex_t v_id;
	offset =0;
	next = 0;
	curr = 0;
	
	printf("Constructing CSR...\n");
	while(offset<line_count){
		char* sss=ss+curr;
		index = atol(sss)-v_min;
		while((ss[next]!=' ')&&(ss[next]!='\n')&&(ss[next]!='\t')){
			next++;
		}
		while((ss[next]==' ')||(ss[next]=='\n')||(ss[next]=='\t')){
			next++;
		}
		curr = next;

		char* sss1=ss+curr;
		v_id = atol(sss1)-v_min;
		adj[begin[index]+degree[index]] = v_id;
		if(is_reverse) adj[begin[v_id]+degree[v_id]] = index;
		
		//Added by Hang
		int rand_weight=(rand()%63+1);
		weight[begin[index]+degree[index]] = rand_weight;
		if(is_reverse)
			weight[begin[v_id]+degree[v_id]] = rand_weight;
		//-End
	
		head[begin[index]+degree[index]]= index;
		while((ss[next]!=' ')&&(ss[next]!='\n')&&(ss[next]!='\t')){
			next++;
		}
		while((ss[next]==' ')||(ss[next]=='\n')||(ss[next]=='\t')){
			next++;
		}
		curr = next;
		degree[index]++;
		if(is_reverse) degree[v_id]++;

		offset++;
	}
	
	//step 5
	//print output as a test
//	for(size_t i=0; i<vert_count; i++){
	for(size_t i=0; i<(vert_count<8?vert_count:8); i++){
		cout<<i<<" "<<begin[i+1]-begin[i]<<" ";
		for(index_t j=begin[i]; j<begin[i+1]; j++){
			cout<<adj[j]<<" ";
		}
//		if(degree[i]>0){
			cout<<endl;
//		}
	}

//	for(int i=0; i<edge_count; i++){
//	for(int i=0; i<64; i++){
//		cout<<head[i]<<"	"<<adj[i]<<endl;
//	}

	munmap( ss,sizeof(char)*file_size );
	
	//-Added by Hang
	munmap( weight,sizeof(vertex_t)*edge_count );
	//-End

	munmap( adj,sizeof(vertex_t)*edge_count );
	munmap( head,sizeof(vertex_t)*edge_count );
	munmap( begin,sizeof(index_t)*vert_count+1 );
	munmap( degree,sizeof(index_t)*vert_count );
	close(fd2);
	close(fd3);
	close(fd4);
	close(fd5);
	
	//-Added by Hang
	close(fd6);
	//-End
}
