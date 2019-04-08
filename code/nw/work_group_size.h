#ifndef WORK_GROUP_SIZE_H_
#define WORK_GROUP_SIZE_H_

#ifndef BSIZE
#ifdef RD_WG_SIZE_0_0
	#define BSIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
	#define BSIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
	#define BSIZE RD_WG_SIZE
#else
	#define BSIZE 128 
	//#define BSIZE 4096 
	//#define BSIZE 8192 
#endif 
#endif // BSIZE

#ifndef PAR
	//#define PAR 64
	//#define PAR 48 
	//#define PAR 32 
	//#define PAR 16 
	#define PAR 8 
#endif

#endif 
