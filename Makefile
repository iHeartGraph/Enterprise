EXE=enterprise.bin

COMMFLAGS=-O3 --compiler-options -Wall -Xptxas -v
CUCC= "$(shell which nvcc)"

CUFLAGS= -arch=sm_35  ${COMMFLAGS}#-Xptxas -dlcm=cg#disable l1 cache
CUFLAGS+= -ccbin=g++ -Xcompiler -fopenmp

ifeq ($(enable_monitor), 1)
	CUFLAGS+= -DENABLE_MONITORING
endif

ifeq ($(enable_check), 1)
	CUFLAGS+= -DENABLE_CHECKING
endif


OBJS=  	main.o 
DEPS= 	Makefile \
		expander.cuh \
		inspector.cuh \
		comm.h \
		bfs_gpu_opt.cuh \
		wtime.h \
		write_result.cuh \
		scan.cuh \
		allocator.cuh 

%.o:%.cu $(DEPS)
	${CUCC} -c  ${CUFLAGS} $< -o $@

${EXE}:${OBJS}
	${CUCC} ${OBJS} $(CUFLAGS) -o ${EXE}

clean:
	rm -rf *.o ${EXE}
