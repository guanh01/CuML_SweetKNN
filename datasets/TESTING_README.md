# Testing Instructions
This directory is organized by datasets, with individual test files located within the dataset's test directory. Each test file follows the general invokation pattern of:  

    python3 test_file_name.py q_landmarks t_landmarks k point_to_display

unless otherwise stated. All test scripts should be run from the directory in which they reside.

## 3D_spatial network
### test_3d.py

    python3 test_3d.py 150 150 20 10

Runs Sweet KNN and brute force KNN with the same settings and compares distance results to determine point matching accuracy.

## arcene
This dataset was not evaluated.

## blogData
### test_blogData.py

    python3 test_blogData.py 150 150 20 10

Runs Sweet KNN and brute force KNN with the same settings and compares distance results to determine point matching accuracy. Crashes during Sweet KNN algorithm due to memory access bug referenced in report.

## dorothea
This dataset was not evaluated.

## ipums
## test_ipums.py

    python3 test_ipums.py 800 800 20 100

Runs Sweet KNN and brute force KNN with the same settings and compares distance results to determine point matching accuracy. Crashes during Sweet KNN algorithm due to memory access bug referenced in report.

## kddcup99
This dataset was not evaluated.

## kegg
### test_kegg.py

    python3 test_kegg.py 800 800 20 100

Runs Sweet KNN and brute force KNN with the same settings and compares distance results to determine point matching accuracy. Crashes during Sweet KNN algorithm due to memory access bug referenced in report.

## keggd
## test_keggd.py

    python3 test_keggd.py 800 800 20 100

Runs Sweet KNN and brute force KNN with the same settings and compares distance results to determine point matching accuracy.

## skin
### test_skin.py

    python3 test_skin.py 150 150 31 100

Runs Sweet KNN and brute force KNN with the same settings and compares distance results to determine point matching accuracy.

### run_tests.sh

    ./run_tests.sh

Invokes the test_skin.py script above to run the script across a range of landmark values. Used to generate data for graphs in project report.

### test_cudf_vs_ndarray.py

    python3 test_cudf_vs_ndarray.py 800 800 20 100

Tests speed of using ndarray vs cuDF dataframe.

### get_misses_for_points.py

    get_misses_for_points.py 150 150 31 100

Run Sweet KNN and brute force KNN for the same k value, display output for all unmatching points. **NOTE** If the number of unmatched points for the selected landmark values is high, this will produce a lot of output. It is best to select values for the landmarks that yields a lower number of unmatched points for comparison. The values in the example command should yield 6 unmatching points, which was used for the report.