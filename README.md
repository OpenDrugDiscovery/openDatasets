# Datasets

## Data Sources & Download Links

| Data Source             | Download Link                                                                        |
|-------------------------|--------------------------------------------------------------------------------------|
| mcule                   | https://mcule.s3.amazonaws.com/database/mcule_purchasable_full_230323.smi.gz         |
| Pubchem                 | https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/                      |
| ChEMBL                  | https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_32_chemreps.txt.gz |
| Zinc (Clean, Drug-like) | http://zinc12.docking.org/db/bysubset/13/usual.sdf.csh                               |

## Required Dependencies
1. gcc and g++
2. cmake
3. boost
4. RDKit

## Running the Code
To build and run the `proc_chembl.cpp` program, do the following:

```
mkdir build
cd build

cmake ..
make
```
You can now run the program on the ChEMBL dataset. The accepted command line usage is shown below.
```
./proc_chembl --help
Usage: ./proc_chembl
  -i/--input <path to input .tsv file>
  -o/--output <path to output .tsv file>
  -m/--max-mol-weight <max mol weight to process> (default: 1000 dA)
  -f/--max-frag-weight <max frag weight to process> (default: 300 dA)
```