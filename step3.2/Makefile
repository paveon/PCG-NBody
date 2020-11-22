# Paralelní programování na GPU (PCG 2020)
# Projekt c. 1 (cuda)
# Login: xpavel34


N=4096
#N=16384
DT=0.01f
STEPS=500
THREADS_PER_BLOCK=1024
#THREADS_PER_BLOCK=256
RED_THREADS=4096
RED_THREADS_PER_BLOCK=128
WRITE_INTESITY=20

INPUT=../sampledata/sampleInput.h5
#INPUT=../32K.h5
OUTPUT=testOutput.h5

INCLUDE=../commons
LIBS_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial/
LIBS=-lhdf5

FLAGS=-Xptxas -v
#FLAGS=

.PHONY: all clean run profile

all: nbody

nbody: nbody.cu main.cu nbody.h
	nvcc ${FLAGS} -I${INCLUDE} -I/usr/include/hdf5/serial nbody.cu main.cu -L${LIBS_DIR} ../commons/h5Helper.cpp ${LIBS} -o nbody
	

clean:
	rm -f *.o nbody

run:
	./nbody ${N} ${DT} ${STEPS} ${THREADS_PER_BLOCK} ${WRITE_INTESITY} ${RED_THREADS} ${RED_THREADS_PER_BLOCK} $(INPUT) $(OUTPUT)
	
my_profile:
	nvprof --print-gpu-trace \
		--devices 0 \
		--metrics flops_sp \
		--metrics flops_sp_special \
		--metrics flop_sp_efficiency \
		--metrics sm_efficiency \
		--metrics achieved_occupancy \
		--metrics alu_fu_utilization \
		--metrics gld_transactions \
		--metrics l1_cache_global_hit_rate \
		./nbody ${N} ${DT} 1 ${THREADS_PER_BLOCK} 0  ${RED_THREADS} ${RED_THREADS_PER_BLOCK} $(INPUT) $(OUTPUT) 

profile:
	nvprof --print-gpu-trace \
		--devices 0 \
		--metrics flops_sp \
		--metrics flops_sp_add \
		--metrics flops_sp_mul \
		--metrics flops_sp_fma \
		--metrics flops_sp_special \
		--metrics flop_sp_efficiency \
		--metrics sm_efficiency \
		--metrics achieved_occupancy \
		--metrics alu_fu_utilization \
		--metrics gld_transactions \
		--metrics l1_cache_global_hit_rate \
		--metrics shared_load_transactions \
		./nbody ${N} ${DT} 1 ${THREADS_PER_BLOCK} 0  ${RED_THREADS} ${RED_THREADS_PER_BLOCK} $(INPUT) $(OUTPUT) 
