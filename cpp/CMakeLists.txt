cmake_minimum_required(VERSION 3.18)

project(proc_chem_db)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(RDKit REQUIRED)
find_package(Boost COMPONENTS system REQUIRED)
add_executable(proc_chembl proc_chembl.cpp)
target_link_libraries(proc_chembl RDKit::SmilesParse RDKit::MolHash)

