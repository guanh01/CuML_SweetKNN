# Improved Sweet KNN
@By [Guoyang Chen][1](gychen1991@gmail.com)
Improved by Ben Demick (bjdemick@ncsu.edu)

## Improvements
The command line arguments have been improved to take the desired point to display as 
the last argument on the command line to facilitate easier testing.

## Prerequisites:
1. NVIDIA GPU card installed on machine.
2. CUDA drivers and CUDA toolkit installed(See https://developer.nvidia.com/cuda-toolkit)

## To compile:
	make

## To Run Sample Script:
	./run_sample.sh (find 200 nearest neighbors)
	
## Usage
	./knnjoin num_query_points num_target_points dimension q_landmarks t_landmarks k query_file target_file point_to_display

## Memory Corruption Reproduction
This command will cause the program to experience a Misaligned Access Exception for the ipums dataset:
	./knnjoin 8844 8844 61 150 150 31 test/s.ipums.la.99.new test/s.ipums.la.99.new 100

## Point difference comparisons
To compare results against the brute force algorithm, run the Python implementation for the desired parameters and observe the output to determine which points disagree. Then run the below command (ensuring that the landmark and k parameters are the same as the Python test), substituting each point from the Python test as the last parameter to the command:
	./knnjoin 245057 245057 4 150 150 31 test/Skin_NonSkin.txt test/Skin_NonSkin.txt 62675