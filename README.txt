README.txt - generated 20190408 by Anthony M. Cabrera and Roger D. Chamberlain

This document describes the contents of research artifacts  
described in the paper:

    Anthony M. Cabrera and Roger D. Chamberlain. 2019.
    Exploring Portability and Performance of OpenCL FPGA Kernels on Intel
    HARPv2. In IWOCL'19: ACM International Workshop on OpenCL.
    May 13-15, 2019, Boston, Massachusetts, USA. DOI: 10.1145/3318170.3318180

-------------------
General Information
-------------------

1. Title - Exploring Portability and Performance of OpenCL FPGA Kernels on
      Intel HARPv2: Research Artifacts 

2. Author Information

   Principal Investigator:
      Roger D. Chamberlain
      Dept. of Computer Science and Engineering
      Washington University in St. Louis
      roger@wustl.edu

   Lead Author:
      Anthony M. Cabrera
      Dept. of Computer Science and Engineering
      Washington University in St. Louis
      acabrera@wustl.edu

3. Date - These artifacts were created during 2018 and 2019.

4. Geographic Location - This research took place in St. Louis, MO, USA.

5. Funding Sources - This work was supported by NSF grants CNS-1205721,
	 CSR-1527510, CCF-1527692, and CNS-1763503.  

--------------------------
Sharing/Access Information
--------------------------

1. License - The data set is in the public domain. Some of the constituent
   software is distributed under a free software license (granting rights
   to modify and redistribute). Details of those licenses are included
   with the specific applications for which they pertain.

2. Publication - The paper is:

   Anthony M. Cabrera and Roger D. Chamberlain. 2019.
   Exploring Portability and Performance of OpenCL FPGA Kernels on Intel
   HARPv2. In IWOCL'19: ACM International Workshop on OpenCL.
   May 13-15, 2019, Boston, Massachusetts, USA. DOI: 10.1145/3318170.3318180

3. Links to other locations - A copy and any updated versions of this data
   are available at https://github.com/cabreraam/iwocl2019_artifacts

4. Links to ancillary data sets - none.

5. Original sources for input data - none.


--------------------
Data & File Overview
--------------------

This is version 1 of the artifacts. 

There are 129 files total, organized in the current directory and 54 
sub-directories. 

In the current directory:

	- README.txt - this file
	- code - the code kernels used in the project
	- data - all of the data used to produce the graphs and figures in the paper

The code directory includes the host and device code as well as Makefiles.
The code directory has 3 sub directories:

	- common - common OpenCL headers and makefile information
	- common_harp - common OpenCL headers and makefile information specific to
	  	HARPv2
	- nw - source code, kernels, and run scripts for the application

The data directory has 3 sub directories: 

	- all_acl_quartus_reports  - contains FPGA resource utilization data
	- timing_data - contains all execution time data
	- compiletime_data - contains a log of the time taken to build kernels

--------------------------
Methodological Information
--------------------------

1. Source - The kernels and data collection methods are described in the
   publication.

2. Processing - The data processing methods are described in the publication. 

3. Personnel - All data handling and analysis was performed by the authors.
