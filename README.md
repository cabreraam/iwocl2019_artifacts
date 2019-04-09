
# Exploring Portability and Performance of OpenCL FPGA Kernels on Intel HARPv2 
Exploring Portability and Performance of OpenCL FPGA Kernels on Intel HARPv2 

This repository contains the most recent version of the artifacts (code and 
data) associated with our IWOCL'19 paper. 

The snapshot of the code at the time of publication is hosted at [WU 
OpenScholarship](https://doi.org/10.7936/m2yq-a123). 

The work done here extends the 
[work of Zohouri et al.](https://github.com/zohourih/rodinia_fpga/) in creating 
the initial port of a subset of the 
[Rodinia benchmarking suite](https://rodinia.cs.virginia.edu/doku.php) 
from GPU-centric implementations to FPGA-centric ones.

## Build Information 
To build the code and kernels, it is assumed that you have access to a HARPv2
node that supports OpenCL and the Intel FPGA SDK/`aoc` offline compiler. If  
this is not the case, you must find access to one.

First, create a folder named `bin_fpga`. This is where the host code will look
for FPGA .aocx files.

To build kernel Version 5, issue the command:

```
aoc nw_kernel_v5.cl -o bin_fpga/nw_kernel_v5_${bsize}_${par}.aocx \
  --board bdw_fpga_v1.0 -v --report -g -DBSIZE=${bsize} -DPAR=${par}

```

where `bsize` and `par` set the `BSIZE` and `PAR` parameters that you wish to
use. 


To build kernel Version 0, issue the command:

```  
aoc nw_kernel_v0.cl -o bin_fpga/nw_kernel_v0_${bsize}.aocx \
  --board bdw_fpga_v1.0 -v --report -g -DBSIZE=${bsize} 

```

where `bsize` sets the `BSIZE` parameter that you wish to use.

To build kernel Version 2, use the same command as Version 0, except change 
**all** instances of  `nw_kernel_v0` to `nw_kernel_v2`.


To build kernel Version 1, issue the command:

```
aoc nw_kernel_v1.cl -o bin_fpga/nw_kernel_v1.aocx \
  --board bdw_fpga_v1.0 -v --report -g 

```

To build kernel Version 3, use the same command as Version 1, except change 
**all** instances of  `nw_kernel_v1` to `nw_kernel_v3`.

To builld the host code for a given kernel, you must define the `BSIZE` and/or
`PAR` parameters for `make` when applicable. For example, to build host code
for kernel Version 5, issue the following command

```
make ALTERA=1 BSIZE=<BSIZE_VAL> PAR=<PAR_VAL>
```

where `BSIZE_VAL` and `PAR_VAL` are set to the same values of `bsize` and `par`
that were used earlier when building the Version 5 kernel.

*NOTE*: defining `ALTERA=1` is required for all host code builds, regardless of
whether `BSIZE` and/or `PAR` are applicable.

To build the host code with SVM enabled, use the command:

```
make ALTERA=1 BSIZE=<BSIZE_VAL> PAR=<PAR_VAL> -f Makefile.svm
```

The only difference is the `Makefile` in this case.


## Running the Application

Once the host and device are built, the [`run`](run) and [`run_svm`](svm) 
scripts can be used to run the applications for without and with, respectively, 
the SVM feature. We will look at the `run` script below. The only difference is
which version of the host code will be used.

```
version=$1
./nw_harp_test 23040 10 1 ${version} 
```

The script takes the kernel version (i.e., `v0`, `v1`, `v2`, `v3`, or `v5`) as a
command-line argument.

The arguments to the application binary are the (square size) of the NW
substitution matrix (which must be a multiple of 16), the gap penalty, a 1 or 0
to turn on/off algorithm verification (respectively), and what version of the
kernel will be used. The last argument is set by the input to the run script.

## Publications

This repository is associated with the following publications.

- Anthony M. Cabrera and Roger D. Chamberlain, "Exploring Portability and 
  Performance of OpenCL FPGA Kernels on Intel HARPv2", in *IWOCL'19*: ACM 
  International Workshop on OpenCL", May 2019.
doi:[10.1145/3318170.3318180](https://www.iwocl.org/iwocl-2019/conference-program/)

- Anthony M. Cabrera and Roger D. Chamberlain, "Exploring Portability and 
  Performance of OpenCL FPGA Kernels on Intel HARPv2: Research Artifacts", in  
  WashU OpenScholarship", April 2019.
  doi:[10.7936/m2yq-a123](https://www.iwocl.org/iwocl-2019/conference-program/)

To cite this work, you can use the following BibTeX entry
```
@inproceedings{cabrera2019exploring,
  title={Exploring Portability and Performance of OpenCL FPGA Kernels on Intel
HARPv2.},
  author={Anthony M. Cabrera and Roger D. Chamberlain},
  booktitle={IWOCL},
  year={2019}
}
```

## Contact

Anthony Cabrera<br />
Department of Computer Science and Engineering<br />
McKelvey School of Engineering<br />
Washington University in St. Louis<br />
firstinitiallastname at wustl dot edu<br />

Roger Chamberlain<br />
Department of Computer Science and Engineering<br />
McKelvey School of Engineering<br />
Washington University in St. Louis<br />
firstname at wustl dot edu<br />

