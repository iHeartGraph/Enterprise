--
Software requirement
-----
gcc 4.4.7 or 4.8.5

CUDA 5.5, 6.0, 6.5, 7.0, 7.5 (tested and works)

--
Hardware
------
GPU: C2050, C2070, K20, K40 (tested)

--
Compile
-----

make

--
Execute
------
Type: "./enterprise.bin" it will show you what is needed.

Tips: It needs a CSR formated graph (beg file and csr file). 

You could use the code from "tuple_text_to_bin.binary_csr" folder to convert a edge list (text formated) graph into CSR files (binary), e.g., if you have a text file called "test.txt", it will convert it into "test.txt_beg_pos.bin" and "test.txt_csr.bin". You will need these two files to run enterprise.

--
Converter: edge tuples to CSR
----
- Compile: make
- To execute: type "./text_to_bin.bin", it will show you what is needed
- Basically, you could download a tuple list file from [snap](https://snap.stanford.edu/data/). Afterwards, you could use this converter to convert the edge list into CSR format. 

**For example**:

- Download https://snap.stanford.edu/data/com-Orkut.html file. **unzip** it. 
- **./text_to_bin.bin soc-orkut.mtx 1 2(could change, depends on the number of lines are not edges)**
- You will get *soc-orkut.mtx_beg_pos.bin* and *soc-orkut.mtx_csr.bin*. 
- You could use these two files to run enterprise.

--
Code specification
---------
The overall code structure of this project is:

- main.cu: main function.

- graph.hpp,graph.h: read the csr format graph file.

- allocator.cuh: alloc GPU memory and copy the graph from CPU to GPU.

- bfs_gpu_opt.cuh: run enterprise BFS.

 - expander.cuh: explore the neighbors of each frontier and mark next level frontiers in status array, leverages hub vertex cache and workload balancing as well.

 - inspector.cuh: scan status array to generate frontier queue, also prepare next level hub vertex for caching.

 - scan.cuh: compute offsets of different thread bins in order to put them in global frontier queue in parallel.

- write_result.cuh: write result out for validation.

- validate.h: graph500 alike validate process.

- wtime.h: timing.

- comm.h: common headers or preprocessors shared by all files.

- gather.h: copy results from GPU to CPU. 

**Should you have any questions or interest about this project, please contact me by hang.gwu@gmail.com. I am more than happy to help you :)**

--
Reference
-------
[SC '15] Enterprise: Breadth-First Graph Traversal on GPUs [[PDF](http://home.gwu.edu/~asherliu/publication/enterprise_sc15.pdf)] [[Slides](http://home.gwu.edu/~asherliu/publication/enterprise_sc15.pdf)] [[Blog](http://home.gwu.edu/~asherliu/publication/enterprise_sc15.pdf)]

[SIGMOD '16] iBFS: Concurrent Breadth-First Search on GPUs [[PDF](http://home.gwu.edu/~asherliu/publication/ibfs.pdf)] [[Slides](http://home.gwu.edu/~asherliu/publication/ibfs_slides.pdf)] [[Poster](http://home.gwu.edu/~asherliu/publication/ibfs_poster.pdf)]
