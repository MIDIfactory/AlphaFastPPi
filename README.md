# AlphaFastPPi

<p align="center"><img src="https://github.com/MIDIfactory/AlphaFastPPi/blob/e31e28a4de963a4f5e61c2fab708c153a77e95c3/AlphaPPI.png" height="256" /></p>
AlphaFastPPi is a Python package designed to streamline large-scale protein-protein interaction analysis using AlphaFold-Multimer. For each protein combination tested, AlphaFastPPi will return a single model.

## **Now the same result can be achieved using [AlphaPulldown](https://github.com/KosinskiLab/AlphaPulldown)**
To obtain a single model in the pulldown version of AlphaPulldown v.1.0.4 use options --num_predictions_per_model=1, --model_names=model_1_multimer_v3, --num_cycle=1, --nopair_ms when running run_multimer_job.py.

Note that slightly different results can be achieved if a different model_name is used.



## Requirments 
- AlphaFastPPi requires the Alphafold databases. 
If you are using an HPC system (or any system where you lack admin privileges), you can access detailed instructions for downloading the Alphafold databases [here](https://github.com/kalininalab/alphafold_non_docker). The databases are available in two sizes: full (~ 2.62 TB) and reduced (~ 820 GB)
```
git clone https://github.com/kalininalab/alphafold_non_docker
cd alphafold_non_docker
./download_db.sh -d /absolute/path/to/the/AF2/download/directory
```
- [conda](https://docs.anaconda.com/free/miniconda/miniconda-install/) 

## Installation

- Create the AlphaFastPPi environment, gathering necessary dependencies:

```
conda create -n AlphaFastPPi -c omnia -c bioconda -c conda-forge python==3.10 openmm==8.0 pdbfixer==1.9 kalign2 cctbx-base pytest importlib_metadata hhsuite
```
- Activate the AlphaFastPPi enviorment and install AlphaPulldown:
```
conda activate AlphaFastPPi
python3 -m pip install alphapulldown==1.0.4
```
- Clone AlphaFastPPi repository
```
git clone https://github.com/MIDIfactory/AlphaFastPPi.git
```
- To utilize GPUs, which significantly accelerate processing though are not strictly necessary, CUDA should be installed:
```
pip install jax==0.4.23 jaxlib==0.4.23+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html 
pip install tensorflow[and-cuda]
conda install -c anaconda cudnn
conda install -c conda-forge cuda-toolkit==11
conda install -c nvidia cuda-nvcc
```
CUDA version can change accordingly to your system's GPU, operating system and the actual version of the software. **If CUDA is not properly installed, the script will use CPU resources automatically**.

## Usage

AlphaFastPPi supports two different modes:
- **pulldown**: to screen a list of proteins ("baits") against a list of other proteins ("candidates")
- **all_vs_all**: to model all pairs of a protein list



1. **Create the MSAs using AlphaPulldown**, compute and store the necessary features for each protein:
```
conda activate AlphaFastPPi
create_individual_features.py \
  --fasta_paths=<fasta file containg all the bait(s) and candidates sequences> \
  --data_dir=<path to alphafold databases> \
  --output_dir=<dir to save the output objects> \ 
  --max_template_date=<any date you want, format like: 2050-01-01> \
  --use_mmseqs2=True
```
` --fasta_paths`: you can use a single fasta file containing all the sequences to include in the analysis or several fasta files separated by comma (e.g. --fasta_paths=protein_A.fasta, protein_B.fasta). \
N.B= the FASTA file should not contain any special characters (such as |, :, ;, #) or spaces. To prevent errors, replace these characters with underscore

`--use_mmseqs2`: when set to "True," mmseqs is executed remotely, which is a **quick** option and typically takes a few minutes per protein. Alternatively, you can set it to "False" to use HHblits locally, or you can run MMseqs locally and then indicate the folder containing the output using the `--use_precomputed_msas` option
\
\
This will create an `--output_dir` formatted like this:
```
output_dir
    |-protein_A.a3m
    |-protein_A_env/
    |-protein_A.pkl
    |-protein_B.a3m
    |-protein_B_env/
    |-protein_B.pkl
    ...
```
2. **Predict the models**:
```
python3 AlphaFastPPi.py 
    --mode <pulldown|all_vs_all>  
    -l proteins.txt \
    -b baits.txt [only for pulldown mode] \
    -d <path to alphafold databases> 
    -m <path to monomer objects dir> \
    -o <name of the output directory>

```
`--mode`: can be **pulldown** or **all_vs_all**
\
\
`-l`: the file should contain a list of the sequences to use (one per line). The names should  match the names of the sequences in the original FASTA file (and in  --monomer_objects_dir). In **pulldown** mode this file should contain **only** the list of the sequences to use as **'candidates'**, while the 'baits' should be listed in another file specified with `-b`. Both files should be formatted as follows:

```
protein_A
protein_B
...
```
\
`-m`: Path to the output_dir created by *create_individual_features.py* 

## Output
For each protein-protein combination, the output will include a subfolder named proteinA_and_proteinB which contains the following files:
- the model in .pdb format
- the corresponding .pkl file
- timings.json

Additionally, a table named output_name.tsv will be generated, containing the following metrics:
- [pDockQ](https://doi.org/10.1038/s41467-022-28865-w)
- ipTM
- ipTM+pTM
- Average plDDT

## Example - Pull down experiment
Input example files are provided in the "example" folder of this repository

1. Compute the MSA using mmseq2
```
conda activate AlphaFastPPi
create_individual_features.py \
  --fasta_paths=bait.fasta, candidates.fasta \
  --data_dir=/mnt/datadisk/AlphaFoldDBs \
  --output_dir=1_fastmsa \ 
  --max_template_date=2050-01-01 \
  --use_mmseqs2=True
```
2. Structural predictions
```
python3 AlphaFastPPi.py 
    --mode pulldown  
    -l candidates.txt \
    -b bait.txt \
    -d /mnt/datadisk/AlphaFoldDBs 
    -m 1_fastmsa \
    -o 2_predictions
```



## Citation

If our tool is useful to you, please cite:
- Bellinzona G, Sassera D, Bonvin AMJJ. Accelerating Protein-Protein Interaction screens with reduced AlphaFold-Multimer sampling. bioRxiv 2024.06.07.597882; doi: https://doi.org/10.1101/2024.06.07.597882
- Yu D, Chojnowski G, Rosenthal M, Kosinski J. AlphaPulldown-a python package for protein-protein interaction screens using AlphaFold-Multimer. Bioinformatics. 2023;39(1):btac749. doi:10.1093/bioinformatics/btac749 


