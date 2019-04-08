include ../common/make.config
include ../common/make.harp

MAIN = nw
KERNEL = $(MAIN)_kernel

ifeq ($(OS),Windows_NT)
	EXE = $(MAIN).exe
else
	EXE = $(MAIN)_harp_test
endif

ifdef BSIZE
	CXXFLAGS += -DBSIZE=$(BSIZE)
endif

ifdef PAR
	CXXFLAGS += -DPAR=$(PAR)
endif

ifdef OUTPUT 
	CXXFLAGS += -DOUTPUT=1
endif

ifdef NO_INTERLEAVE
	$(ECHO) CXXFLAGS += -DNO_INTERLEAVE
endif

all: $(MAIN)_harp_test.c 
	$(ECHO) $(CXX) -fPIC $(foreach D, $(INC_DIRS), -I$D) \
		$(AOCL_COMPILE_CONFIG) $(SRCS) $(AOCL_LINK_CONFIG) \
		$(foreach D,$(LIB_DIRS),-L$D) \
		$(foreach L,$(LIBS),-l$L) \
		${KERNEL_DIM} $(CXXFLAGS) $(MAIN)_harp_test.c $(OPENCL_INC) $(OPENCL_LIB) -o $(EXE)
#	$(CXX) ${KERNEL_DIM} $(CXXFLAGS) $(MAIN).c $(OPENCL_INC) $(OPENCL_LIB) -o $(EXE)
	
verify: $(MAIN)_harp_test.c 
	$(ECHO) $(CXX) -fPIC $(foreach D, $(INC_DIRS), -I$D) \
		$(AOCL_COMPILE_CONFIG) $(SRCS) $(AOCL_LINK_CONFIG) \
		$(foreach D,$(LIB_DIRS),-L$D) \
		$(foreach L,$(LIBS),-l$L) \
		${KERNEL_DIM} $(CXXFLAGS) $(MAIN)_harp_test.c $(OPENCL_INC) $(OPENCL_LIB) -o $(EXE) -DVERIFY -DNO_INTERLEAVE
	#echo "${NO_INTERLEAVE} hello"
#	$(CXX) ${KERNEL_DIM} $(CXXFLAGS) $(MAIN).c $(OPENCL_INC) $(OPENCL_LIB) -o $(EXE) -DVERIFY

trace: $(MAIN).c
	${ECHO} $(CXX) ${KERNEL_DIM} $(CXXFLAGS) $(MAIN).c $(OPENCL_INC) $(OPENCL_LIB) -o $(EXE) -DTRACEBACK
#	$(CXX) ${KERNEL_DIM} $(CXXFLAGS) $(MAIN).c $(OPENCL_INC) $(OPENCL_LIB) -o $(EXE) -DTRACEBACK

ifndef HOST_ONLY
all: kernel
trace: kernel
endif

kernel:

ifdef ALTERA
# default version
#v ?= 9
# Anthony note: changed default to 1
v ?= 1 
# default block size
BSIZE ?= 128
# default SIMD size
SIMD ?= 4

ifdef asdkfjasdklfjsal
# version
KERNEL_VER = $(KERNEL)_v$(v)
KERNEL_VER_AOCO = $(KERNEL_VER)$(PROFILE_SUFFIX).aoco
KERNEL_VER_AOCX = $(KERNEL_VER)$(PROFILE_SUFFIX).aocx
$(KERNEL_VER_AOCO): $(KERNEL_VER).cl
	$(RM) $(KERNEL_VER)$(PROFILE_SUFFIX)
	$(CFPGA) -c $< $(CFPGA_FLAGS) -o $@

# BSIZE
KERNEL_VER_BSIZE = $(KERNEL_VER)_BSIZE$(BSIZE)
KERNEL_VER_BSIZE_AOCO = $(KERNEL_VER_BSIZE)$(PROFILE_SUFFIX).aoco
KERNEL_VER_BSIZE_AOCX = $(KERNEL_VER_BSIZE)$(PROFILE_SUFFIX).aocx
$(KERNEL_VER_BSIZE_AOCO): DEFINE_MACROS += -DBSIZE=$(BSIZE)
$(KERNEL_VER_BSIZE_AOCO): $(KERNEL_VER).cl
	$(RM) $(KERNEL_VER_BSIZE)$(PROFILE_SUFFIX)
	$(CFPGA) -c $< $(CFPGA_FLAGS) -o $@

# SIMD
KERNEL_VER_BSIZE_SIMD = $(KERNEL_VER_BSIZE)_SIMD$(SIMD)
KERNEL_VER_BSIZE_SIMD_AOCO = $(KERNEL_VER_BSIZE_SIMD)$(PROFILE_SUFFIX).aoco
KERNEL_VER_BSIZE_SIMD_AOCX = $(KERNEL_VER_BSIZE_SIMD)$(PROFILE_SUFFIX).aocx
$(KERNEL_VER_BSIZE_SIMD_AOCO): DEFINE_MACROS += -DBSIZE=$(BSIZE) -DSIMD=$(SIMD)
$(KERNEL_VER_BSIZE_SIMD_AOCO): $(KERNEL_VER).cl
	$(RM) $(KERNEL_VER_BSIZE_SIMD)$(PROFILE_SUFFIX)
	$(CFPGA) -c $< $(CFPGA_FLAGS) -o $@


# target definitions
kernel: aocx
ifneq (,$(findstring $(v),1 3))
aocx: $(KERNEL_VER_AOCX)
aoco: $(KERNEL_VER_AOCO)
else ifneq (,$(findstring $(v),0 7 9 11 13 15))
aocx: $(KERNEL_VER_BSIZE_AOCX)
aoco: $(KERNEL_VER_BSIZE_AOCO)
else ifneq (,$(findstring $(v),2 4))
aocx: $(KERNEL_VER_BSIZE_SIMD_AOCX)
aoco: $(KERNEL_VER_BSIZE_SIMD_AOCO)
endif

endif # ifdef asdkfjasdklfjsal

endif # ifdef ALTERA

clean:
	$(RM) $(EXE) *.aoco
