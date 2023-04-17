#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>

#include <boost/algorithm/string.hpp>

#include <GraphMol/MolOps.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/FileParsers/MolWriters.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <GraphMol/FileParsers/MolSupplier.h>
#include <GraphMol/Descriptors/MolDescriptors.h>
#include <GraphMol/ChemTransforms/MolFragmenter.h>

void updateProgBar(float currProg) {
    int barWidth = 70;

    std::cout << "[";
    int pos = barWidth * currProg;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(currProg * 100.0) << " %\r";
    std::cout.flush();

}

int countLines(std::string fname) {
    std::ifstream infile(fname);
    int numLines = std::count(std::istreambuf_iterator<char>(infile), 
                   std::istreambuf_iterator<char>(), '\n');
    infile.close();
    return numLines;
}

void getMolFrags(RDKit::ROMol *mol, std::vector<RDKit::ROMol*> &frags, const int discardGtEq = 300) {
    auto mol_frags = RDKit::MolOps::getMolFrags(*mol, true);
    for (auto &frag : mol_frags) {
        const auto *wrapped_frag = frag.get();
        double avgMolWeight = RDKit::Descriptors::calcAMW(*wrapped_frag);
        if (avgMolWeight < discardGtEq) {
            frags.push_back(new RDKit::RWMol(*wrapped_frag));
        }
    }
}

void process_chembl(std::string in_fname, std::string out_fname, const float maxMolWeight, const float maxFragWeight) {
    int numLines = countLines(in_fname);
    std::cout << "Processing " << in_fname << " with " << numLines << " lines" << std::endl;

    std::ifstream infile(in_fname);
    std::ofstream outfile(out_fname);
    std::string line;

    int i = 0;

    std::string id;
    std::string smiles;
    std::string std_inchi;
    std::string std_inchi_key;

    std::vector<RDKit::ROMol*> frags;

    int update_every = 0.01 * numLines;
    
    auto start = std::chrono::high_resolution_clock::now();
    while (std::getline(infile, line)) {
        i++;
        if (i == 1) {
            // we can skip the header
            outfile << line;
            continue;
        }
        if ((i - 1) % update_every == 0 || i == numLines) {
            float progress = (float) (i - 1) / (float) numLines;
            updateProgBar(progress);
        }
        std::vector<std::string> cols;
        boost::split(cols, line, boost::is_any_of("\t"));
        id = cols[0];
        smiles = cols[1];
        std_inchi = cols[2];
        std_inchi_key = cols[3];

        RDKit::ROMol *mol = RDKit::SmilesToMol(smiles);
        if (RDKit::Descriptors::calcAMW(*mol) > maxMolWeight) {
            // remove mols w/ weight > 1kDa
            continue;
        }
        RDKit::ROMol *fragmented = RDKit::MolFragmenter::fragmentOnBRICSBonds(*mol); // use BRICS algo
        getMolFrags(fragmented, frags, maxFragWeight);
        for (auto &frag : frags) {
            outfile << id << "\t" << RDKit::MolToSmiles(*frag) << "\t" << std_inchi << "\t" << std_inchi_key << "\n";
        }
        frags.clear(); // reset the fragment vector
    }
    // cleanup
    infile.close();
    outfile.close();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << std::endl;
    std::cout << "Processed " << i << " lines in " << duration.count() / 1000.0 << " seconds" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2 || !std::strcmp(argv[1], "-h") || !std::strcmp(argv[1], "--help")) {
        // Tell the user how to run the program
        std::cerr << "Usage: " << argv[0] << std::endl;
        std::cerr << "  -i/--input <path to input .tsv file>" << std::endl;
        std::cerr << "  -o/--output <path to output .tsv file>" << std::endl;
        std::cerr << "  -m/--max-mol-weight <max mol weight to process> (default: 1000 dA)" << std::endl;
        std::cerr << "  -f/--max-frag-weight <max frag weight to process> (default: 300 dA)" << std::endl;
        return 1;
    }
    int i = 1;
    float maxMolWeight = 1000;
    float maxFragWeight = 300;
    std::string in_fname;
    std::string out_fname;
    while (i < argc) {
        std::string arg = argv[i];
        if (arg == "-i" || arg == "--input") {
            in_fname = argv[i + 1];
            i += 2;
        } else if (arg == "-o" || arg == "--output") {
            out_fname = argv[i + 1];
            i += 2;
        } else if (arg == "-m" || arg == "--max-mol-weight") {
            maxMolWeight = std::stof(argv[i + 1]);
            i += 2;
        } else if (arg == "-f" || arg == "--max-frag-weight") {
            maxFragWeight = std::stof(argv[i + 1]);
            i += 2;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return 1;
        }
    }
    if (in_fname == "") {
        std::cerr << "Must provide input file" << std::endl;
        return 1;
    }
    if (out_fname == "") {
        std::cerr << "Must provide output file" << std::endl;
        return 1;
    }
    process_chembl(in_fname, out_fname, maxMolWeight, maxFragWeight);
    return 0;
}

