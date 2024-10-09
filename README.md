# PRIDICT2.0: PRIme editing guide RNA preDICTion 2.0

![PRIDICT logo](dataset/PRIDICT2_logo.jpg)

## Overview

[PRIDICT2.0](https://rdcu.be/dLu0f) is an advanced version of the original [PRIDICT](https://rdcu.be/c3IM5) model designed for predicting the efficiency of prime editing guide RNAs. This repository allows you to run the model locally. For details on advancements over the original model, refer to our published study ([Mathis et al., Nature Biotechnology, 2024](https://rdcu.be/dLu0f)) and the initial [BioRxiv preprint](https://www.biorxiv.org/content/10.1101/2023.10.09.561414v1).

## Complementary Model

- **ePRIDICT**: This model focuses on the influence of local chromatin context (K562) on prime editing efficiencies and is designed to complement PRIDICT2.0. [Access GitHub Repository](https://github.com/Schwank-Lab/epridict)

## Resources

- **Supplementary Files**: [Access Here](https://github.com/Schwank-Lab/epridict/tree/supplementary_files)
- **Web Application**: For an online version of PRIDICT2.0, visit [our webapp](https://pridict.it/).

## Contact

For questions or suggestions, please either:
- Email us at [nicolas.mathis@pharma.uzh.ch](mailto:nicolas.mathis@pharma.uzh.ch)
- Open a GitHub issue

## Citation

If find our work useful for your research please cite:
- [Mathis et al., Nature Biotechnology, 2024](https://rdcu.be/dLu0f) (PRIDICT2.0)
- [Mathis & Allam et al., Nature Biotechnology, 2023](https://rdcu.be/c3IM5) (PRIDICT)



## Getting Started

### Installation using Anaconda (Linux, Mac OS or WSL) 🐍
Windows is only supported via [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install).

The easiest way to install and manage Python packages on various OS platforms is through [Anaconda](https://docs.anaconda.com/anaconda/install/). Once installed, any package (even if not available on Anaconda channel) could be installed using pip. 

* Install [Anaconda](https://docs.anaconda.com/anaconda/install/) or [miniconda](https://docs.anaconda.com/miniconda/). (conda 22.11 or newer)
* Start a terminal and run:
    ```shell
    # clone PRIDICT2.0 repository
    git clone https://github.com/uzh-dqbm-cmi/PRIDICT2.git
    # navigate into repository
    cd PRIDICT2
    # create conda environment and install dependencies for PRIDICT2 (only has to be done before first run/install)
    conda env create -f pridict2_repo.yml
        
    # activate the created environment
    conda activate pridict2

    # pytorch has to be installed separately here:
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
  	
    # run desired PRIDICT2.0 command (single or batch mode, described below)
    python pridict2_pegRNA_design.py single --sequence-name seq1 --sequence "GCCTGGAGGTGTCTGGGTCCCTCCCCCACCCGACTACTTCACTCTCTGTCCTCTCTGCCCAGGAGCCCAGGATGTGCGAGTTCAAGTGGCTACGGCCGA(G/C)GTGCGAGGCCAGCTCGGGGGCACCGTGGAGCTGCCGTGCCACCTGCTGCCACCTGTTCCTGGACTGTACATCTCCCTGGTGACCTGGCAGCGCCCAGATGCACCTGCGAACCACCAGAATGTGGCCGC"
    # results are stored in 'predictions' folder
    ```

* `PRIDICT2.0` environment only has to be installed once. When already installed, follow the following commands to use `PRIDICT2.0` again:
    ```shell
    # open Terminal/Command Line
    # navigate into repository
    # activate the created environment
    conda activate pridict2
    # run desired PRIDICT2.0 command (single or batch mode, described below)
    python pridict2_pegRNA_design.py single --sequence-name seq1 --sequence "GCCTGGAGGTGTCTGGGTCCCTCCCCCACCCGACTACTTCACTCTCTGTCCTCTCTGCCCAGGAGCCCAGGATGTGCGAGTTCAAGTGGCTACGGCCGA(G/C)GTGCGAGGCCAGCTCGGGGGCACCGTGGAGCTGCCGTGCCACCTGCTGCCACCTGTTCCTGGACTGTACATCTCCCTGGTGACCTGGCAGCGCCCAGATGCACCTGCGAACCACCAGAATGTGGCCGC"
    # results are stored in 'predictions' folder
    ```

--------------------------

### Running PRIDICT2.0 in 'single' mode:
  ####  Required:
  -  `--sequence-name`: name of the sequene (i.e. unique id for the sequence)
  -  `--sequence`: target sequence to edit in quotes (format: `"xxxxxxxxx(a/g)xxxxxxxxxx"`; minimum of 100 bases up and downstream of brackets are needed; put unchanged edit-flanking bases *outside* of brackets (e.g. xxxT(a/g)Cxxx instead of xxx(TAC/TGC)xxx)
  ####  Optional:
  -  `--output-dir`: output directory where results are dumped on disk (default: `./predictions`; directory must already exist before running)
  -  `--use_5folds`: Use all 5-folds trained models. Default is to use fold-1 model
  -  `--nicking`: Additionally, design nicking guides for edit (PE3) with DeepSpCas9 prediction ([Kim et al. 2019](https://www.science.org/doi/10.1126/sciadv.aax9249)).
  -  `--ngsprimer`: Additionally, design NGS primers for edit based on [Primer3](https://primer3.org/) design.

Example command:
```shell
python pridict2_pegRNA_design.py single --sequence-name seq1 --sequence "GCCTGGAGGTGTCTGGGTCCCTCCCCCACCCGACTACTTCACTCTCTGTCCTCTCTGCCCAGGAGCCCAGGATGTGCGAGTTCAAGTGGCTACGGCCGA(G/C)GTGCGAGGCCAGCTCGGGGGCACCGTGGAGCTGCCGTGCCACCTGCTGCCACCTGTTCCTGGACTGTACATCTCCCTGGTGACCTGGCAGCGCCCAGATGCACCTGCGAACCACCAGAATGTGGCCGC"
``` 
--------------------------

### Running in batch mode:
For instructions on running PRIDICT2.0 in batch mode on an HPC cluster with SLURM (e.g. with >1000 inputs), see our [HPC Batch Guide](./hpc_batch/README.md).
  ####  Required:
  -  `--input-fname`: input file name - name of csv file that has two columns [`editseq`, `sequence_name`]. See `batch_template.csv` in the `./input` folder
  ####  Optional:
  -  `--input-dir` : directory where the input csv file is found on disk
  -  `--output-dir`: directory on disk where to dump results (default: `./predictions`)
  -  `--use_5folds`: Use all 5-folds trained models. Default is to use fold-1 model
  -  `--cores`: Number of batch input sequences that can be predicted simultaneously. Maximum 3 cores due to memory limitations. Default value 0 uses 3 cores if available.
  -  `--nicking`: Design nicking guides for edit (PE3) with DeepSpCas9 prediction ([Kim et al. 2019](https://www.science.org/doi/10.1126/sciadv.aax9249)).
  -  `--ngsprimer`: Design NGS primers for edit based on [Primer3](https://primer3.org/) design.
  -  `--summarize`: Summarize the best scoring pegRNA(s) of each batch input in a summary file (saved in output folder in the format `date` +`time` + `HEK or K562` + `batch_summary.csv`). Choose either `HEK` or `K562` to get highest scoring pegRNAs based on desired PRIDICT2.0 score. (e.g., `--summarize K562`)
  -  `--summarize_number`: Define the number of top scoring pegRNAs to be included in summary file. Default is `3`.

Example command (including only required arguments):
```shell
 python pridict2_pegRNA_design.py batch --input-fname batch_template.csv
``` 
  #### Notes:
- A log file will be saved in `log` file folder (`date` +`time` + `batch filename (without file extension)` + `_batch_logfile.csv`). Check the column `log` to catch any inputs which contained errors. If everything worked well, it should read `Prediction successful!` in each row.
- To run PRIDICT2.0 from outside the repository folder, add path to script file (e.g. `/mnt/c/Users/pridictuser/github/PRIDICT2/pridict2_pegRNA_design.py`) and use `--input-dir` to define input folder (e.g. `/mnt/c/Users/pridictuser/github/PRIDICT2/input`)

--------------------------

### Prediction of pegRNAs with silent bystander edits

`PRIDICT2.0`'s enhanced ability to predict multi-bp edits allows researchers to improve their outcomes by incorporating silent bystander edits into their targets. These edits can potentially boost editing efficiencies, particularly in MMR proficient contexts.

To facilitate this process, we provide a Jupyter Notebook `notebook_silent_bystander_input.ipynb` in the `silentbystander_addon` folder. This notebook generates all possible `PRIDICT2.0` input sequences with silent bystanders up to 5bp up- and downstream of the edit.

#### Requirements:
- PRIDICT input sequence with 150 bp context on both sides of the edit
- Input sequence must be in-frame (ORF_start = 0) if the edit and its context are within an exon
- If not in an exon, still assume it is "in frame" and also keep the `in frame value` as `yes` if you do batch input file
  --> to get all possible mutations for an intron/non-genic region, enter your input sequence in all ORF variants (3 for the fw strand; 3 for the rv strand) and run the batch mode.
- PRIDICT2.0 conda environment

#### Usage overview:
1. Use the notebook or corresponding command line script to create an input batch file for `PRIDICT2.0` prediction with silent bystanders.
2. Supports 1bp replacements and multi-bp replacements (not insertions/deletions).
3. Input format: `PRIDICT2.0` format with 150bp flanking bases on both sides.
4. Choose between single (single mutation/edit) or batch (multiple inputs) functions.
5. *Run `PRIDICT2.0` in batch mode in a terminal outside of notebook with the generated input sequences to obtain efficiency predictions.
(Optional: run PRIDICT2.0 from within notebook, see commented out example)*
6. Summarize individual bystander predictions to one prediction file (best prediction of each bystander variant)
