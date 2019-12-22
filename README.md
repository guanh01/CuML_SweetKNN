# README for Project 3: Sweet KNN cuML Integration
This folder consists of all the development and test artifacts for Project 3.
## Contents
    README.md # This readme file
    demick_project_3_final_report.pdf
    cuml_patch/
        sweet_patch.diff # Git patch file containing changes to the base git commit
        knn.cu # CUDA KNN implementation containing Sweet KNN changes
        knn.hpp # CUDA KNN header containing Sweet KNN changes
        nearest_neighbors.pyx # Python integration of Sweet KNN

    ImprovedSweetKnn/
        Makefile
        README.md # See this file for instructions on using the improved knnjoin
        run_sample.sh
        src/
            knnjoin.cu
        test/
            Skin_NonSkin.txt
            s.ipums.la.99.new

    datasets/
        TESTING_README.md # See this file for instructions on running tests
        3D_spatial_network/
        arcene/
        blogData/
        dorothea/
        ipums/
        kddcup99/
        kegg/
        keggd/
        skin/
        
## NOTES
There are instructions for reproducing the memory corruption issues in the ImprovedSweetKnn/README.md as well as in datasets/TESTING_README.md.

There is a test script datasets/skin/get_misses_for_points.py for running Sweet and brute algorithms and displaying the specific points that differ between algorithms for each point. Instructions on how to run these tests are located in datasets/TESTING_README.md

## Build Environment Setup and Configuration

### Add to .bashrc
	export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}$
	export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

### Install Anaconda
	wget https://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-x86_64.sh
	chmod +x Anaconda3-5.2.0-Linux-x86_64.sh
	./Anaconda3-5.2.0-Linux-x86_64.sh # Accept the default options
	source ~/.bashrc

### Install the rapidsai suite
	conda install -c rapidsai -c nvidia -c conda-forge -c defaults rapids=0.10 python=3.6
	conda update --all

Log out of the server, and log back in.

### Clone the cuML repository and create the development environment
	git clone https://github.com/rapidsai/cuml.git --branch branch-0.11 --single-branch --recursive
	cd cuml
    git checkout -b sweet 03d8095122cbe101c03a149980c25e9b7b7e73f2
	conda env create -n cuml_dev python=3.6 --file=conda/environments/cuml_dev_cuda10.0.yml

### Build cuML
	source activate cuml_dev # Critical step, this environment must be active to build
    ./build.sh

### Test that build was successful
    python3
    import cuml
If no error messages, build was successful CTRL + D to exit Python3

### Installation of cuML Sweet KNN Patch
    git apply <project_submission_root>/cuml_patch/sweet_patch.diff
    ./build.sh

If patch installation not successful:

Copy:  
    <project_submission_root>t/cuml_patch/knn.hpp to cuml/python/cuml/neighbors/nearest_neighbors.pyx  
    <project_submission_root>t/cuml_patch/knn.hpp to cuml/cpp/include/cuml/neighbors/knn.hpp  
    <project_submission_root>t/cuml_patch/knn.cu to cuml/cpp/src/knn/knn.cu  
Run:  

    ./build.sh

### Run Test Code
See datasets/TESTING_README.md



## Comment

### Contribution of this repo: 
* cleaned up Guoyang's implementation
* mentioned that there are still some bugs need to be fixed as the results are not consistent with the standard KNN in CuML.
* However, according to the other students, Guoyang's code can be executed successfully on all the datasets. There is a slight chance that Ben's optimization to the code breaks Guoyang's code.

### Project description 
cuML is an open-source machine learning library for GPU. It is part of the RAPIDS effort from NVIDIA. The goal of this project is to extend cuML by adding Sweet KNN into it.

Sweet KNN is an optimized K-Nearest Neighbor algorithm. It uses triangle inequality to dramatically improve the computing speed of KNN. Details are shown in this paper (http://people.engr.ncsu.edu/xshen5/Publications/icde17.pdf).

A version of Sweet KNN implemented by the authors of the research paper is available here (https://people.engr.ncsu.edu/xshen5/csc512_fall2019/knn_sweet-master.zip).

### Project requirement 
1) Your extension is to the KNN function in cuML by extending its '__init__' with an extra argument/parameter 'algorithm', whose value can be either "full" (for the default algorithm) or "sweet" for Sweet KNN.

   Try to follow a good coding style. The best submission could be suggested to commit to cuML for the community to use.

2) Try to make the implementation of Sweet KNN as fast as possible.

3) Design and run experiments to compare the performance of the "full" and "sweet" algorithms on the datasets used in the Sweet KNN paper. Note that any performance comparison has to first make sure that the execution results from the algorithms are equivalent. For KNN, the equivalence can be as simple as that the algorithms output the same nearest neighbors.

4) Create the documentations for the extended functionality.

5) Write the final report.

 

