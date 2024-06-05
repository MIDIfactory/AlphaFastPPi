# #
# This script 
# contains code copied from the script run_alphafold.py by DeepMind from https://github.com/deepmind/alphafold
# and predict_structure and run_multiple_jobs.py v.1.04 from https://github.com/KosinskiLab/AlphaPulldown 
# #

# Libraries 
from absl import app, logging
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.relax import relax
from alphafold.model import config
from alphafold.model import model
from alphafold.model import data
from alphapulldown.utils import get_run_alphafold
from alphapulldown.utils import (create_interactors, read_all_proteins, check_output_dir)
from alphapulldown.objects import MultimericObject
from collections import defaultdict
import glob
import itertools
from itertools import combinations
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import random
import sys
import time
import jax.numpy as jnp
import json


# Relaxation
run_af = get_run_alphafold()
RELAX_MAX_ITERATIONS = run_af.RELAX_MAX_ITERATIONS
RELAX_ENERGY_TOLERANCE = run_af.RELAX_ENERGY_TOLERANCE
RELAX_STIFFNESS = run_af.RELAX_STIFFNESS
RELAX_EXCLUDE_RESIDUES = run_af.RELAX_EXCLUDE_RESIDUES
RELAX_MAX_OUTER_ITERATIONS = run_af.RELAX_MAX_OUTER_ITERATIONS

ModelsToRelax = run_af.ModelsToRelax


# Definitions
def _jnp_to_np(output):
  """Recursively changes jax arrays to numpy arrays."""
  for k, v in output.items():
    if isinstance(v, dict):
      output[k] = _jnp_to_np(v)
    elif isinstance(v, jnp.ndarray):
      output[k] = np.array(v)
  return output



def predict(
    model_runners,
    output_dir,
    feature_dict,
    random_seed,
    relaxation_step: ModelsToRelax,
    fasta_name,
    seqs=[],
    use_gpu_relax=True
):
    timings = {}
    unrelaxed_pdbs = {}
    relaxed_pdbs = {}
    relax_metrics = {}
    unrelaxed_proteins = {}
    prediction_result = {}
    START = 0
    temp_timings_output_path = os.path.join(output_dir, "timings_temp.json") 


    num_models = 1
    for model_index, (model_name, model_runner) in enumerate(model_runners.items()):
        if model_index < START:
            continue
        logging.info("Running model %s", fasta_name)
        t_0 = time.time()
        model_random_seed = model_index + random_seed * num_models
        processed_feature_dict = model_runner.process_features(
            feature_dict, random_seed=model_random_seed
        )
        timings[f"process_features_{model_name}"] = time.time() - t_0

        t_0 = time.time()
        prediction_result = model_runner.predict(
            processed_feature_dict, random_seed=model_random_seed
        )

        # update prediction_result with input seqs
        prediction_result.update({"seqs": seqs})

        t_diff = time.time() - t_0
        timings[f"predict_and_compile_{model_name}"] = t_diff
        logging.info(
            "Time required for the prediction: %.1fs",
            t_diff,
        )


        plddt = prediction_result["plddt"]

        # Remove jax dependency from results
        np_prediction_result = _jnp_to_np(dict(prediction_result))

        result_output_path = os.path.join(output_dir, f"result_{fasta_name}.pkl")
        with open(result_output_path, "wb") as f:
            pickle.dump(np_prediction_result, f, protocol=4)

        plddt_b_factors = np.repeat(
            plddt[:, None], residue_constants.atom_type_num, axis=-1
        )
        unrelaxed_protein = protein.from_prediction(
            features=processed_feature_dict,
            result=prediction_result,
            b_factors=plddt_b_factors,
            remove_leading_feature_dimension=not model_runner.multimer_mode,
        )

        unrelaxed_proteins[fasta_name] = unrelaxed_protein
        unrelaxed_pdbs[fasta_name] = protein.to_pdb(unrelaxed_protein)
        unrelaxed_pdb_path = os.path.join(output_dir, f"unrelaxed_{fasta_name}.pdb")
        with open(unrelaxed_pdb_path, "w") as f:
            f.write(unrelaxed_pdbs[fasta_name])

        with open(temp_timings_output_path, "w") as f:
            f.write(json.dumps(timings, indent=4))
        break

    # Relax prediction
    amber_relaxer = relax.AmberRelaxation(
        max_iterations=RELAX_MAX_ITERATIONS,
        tolerance=RELAX_ENERGY_TOLERANCE,
        stiffness=RELAX_STIFFNESS,
        exclude_residues=RELAX_EXCLUDE_RESIDUES,
        max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
        use_gpu=use_gpu_relax)

    if relaxation_step == True:
        to_relax = [fasta_name]
    elif relaxation_step == False:
        to_relax = []

    for fasta_name in to_relax:
        t_0 = time.time()
        relaxed_pdb_str, _, violations = amber_relaxer.process(
            prot=unrelaxed_proteins[fasta_name])
        relax_metrics[fasta_name] = {
            'remaining_violations': violations,
            'remaining_violations_count': sum(violations)
        }
        timings[f'relax_{fasta_name}'] = time.time() - t_0

        relaxed_pdbs[fasta_name] = relaxed_pdb_str

        # Save the relaxed PDB
        relaxed_output_path = os.path.join(
            output_dir, f'relaxed_{fasta_name}.pdb')
        with open(relaxed_output_path, 'w') as f:
            f.write(relaxed_pdb_str)
    
    timings_output_path = os.path.join(output_dir, "timings.json")
    with open(timings_output_path, "w") as f:
        f.write(json.dumps(timings, indent=4))
    if relaxation_step != False:
        relax_metrics_path = os.path.join(output_dir, 'relax_metrics.json')
        with open(relax_metrics_path, 'w') as f:
            f.write(json.dumps(relax_metrics, indent=4))

    if os.path.exists(temp_timings_output_path): 
        try:
            os.remove(temp_timings_output_path)
        except OSError:
            pass

def create_model_runners_and_random_seed(
        model_preset, num_c1ycle, random_seed, data_dir,
        num_multimer_predictions_per_model):
    num_ensemble = 1
    model_runners = {}
    model_names = config.MODEL_PRESETS[model_preset]


    for model_name in model_names:
        model_config = config.model_config(model_name)
        model_config.model.num_ensemble_eval = num_ensemble
        model_config["model"].update({"num_recycle": 1})

        model_params = data.get_model_haiku_params(model_name=model_name, data_dir=data_dir)
        model_runner = model.RunModel(model_config, model_params)


        for i in range(num_multimer_predictions_per_model):
            model_runners[f"{model_name}_pred_{i}"] = model_runner

    if random_seed is None:
        random_seed = random.randrange(sys.maxsize // len(model_runners))
        logging.info("Using random seed %d for the data pipeline", random_seed)

    return model_runners, random_seed



def create_pulldown_info(
        bait_proteins: list, candidate_proteins: list, seq_index=None
) -> dict:
    """
    A function to create apms info

    Args:
    all_proteins: list of all proteins in the fasta file parsed by read_all_proteins()
    bait_protein: name of the bait protein
    seq_index: whether there is a seq_index specified or not
    """
    all_protein_pairs = list(itertools.product(*[bait_proteins, *candidate_proteins]))
    num_cols = len(candidate_proteins) + 1
    data = dict()


    if seq_index is None:
        for i in range(num_cols):
            curr_col = []
            for pair in all_protein_pairs:
                curr_col.append(pair[i])
            update_dict = {f"col_{i + 1}": curr_col}
            data.update(update_dict)


    elif isinstance(seq_index, int):
        target_pair = all_protein_pairs[seq_index - 1]
        for i in range(num_cols):
            update_dict = {f"col_{i + 1}": [target_pair[i]]}
            data.update(update_dict)
    return data



def create_all_vs_all_info(all_proteins: list, seq_index=None):
    """A function to create all against all i.e. every possible pair of interaction"""
    all_possible_pairs = list(combinations(all_proteins, 2))
    if seq_index is not None:
        seq_index = seq_index - 1
        combs = [all_possible_pairs[seq_index-1]]
    else:
        combs = all_possible_pairs


    col1 = []
    col2 = []
    for comb in combs:
        col1.append(comb[0])
        col2.append(comb[1])


    data = {"col1": col1, "col2": col2}
    return data



def create_custom_info(all_proteins):
    """
    A function to create 'data' for custom input file
    """
    num_cols = len(all_proteins)
    data = dict()
    for i in range(num_cols):
        data[f"col_{i+1}"] = [all_proteins[i]]
    return data



def create_multimer_objects(data, monomer_objects_dir, pair_msa=True):
    """
    A function to create multimer objects

    Arg
    data: a dictionary created by create_all_vs_all_info() or create_apms_info()
    monomer_objects_dir: a directory where pre-computed monomer objects are stored (result of create_individual_features.py)
    """
    multimers = []
    num_jobs = len(data[list(data.keys())[0]])
    job_idxes = list(range(num_jobs))
    
    
    pickles = set()
    for path in monomer_objects_dir:
        path = os.path.join(path,'*.pkl')
        pickles.update(set([os.path.basename(fl) for fl in glob.glob(path)]))
        
    required_pickles = set(key+".pkl" for value_list in data.values()
                    for value_dict in value_list
                    for key in value_dict.keys())

    missing_pickles = required_pickles.difference(pickles)
    if len(missing_pickles) > 0:
        raise Exception(f"{missing_pickles} not found in {monomer_objects_dir}. Have you already run create_individual_features.py ?")
    else:
        logging.info("All pickle files have been found")


    for job_idx in job_idxes:
        interactors = create_interactors(data, monomer_objects_dir, job_idx)
        if len(interactors) > 1:
            multimer = MultimericObject(interactors=interactors,pair_msa=pair_msa, multimeric_mode = False)
            logging.info(f"done creating multimer {multimer.description}")
            multimers.append(multimer)
        else:
            logging.info(f"done loading monomer {interactors[0].description}")
            multimers.append(interactors[0])
    return multimers



def predict_individual_jobs(multimer_object, output_path, model_runners, random_seed):
    output_path = os.path.join(output_path, multimer_object.description)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    if not isinstance(multimer_object, MultimericObject):
        multimer_object.input_seqs = [multimer_object.sequence]
    else:
        predict(
            model_runners,
            output_path,
            multimer_object.feature_dict,
            random_seed,
            fasta_name=multimer_object.description,
            relaxation_step=FLAGS.relaxation_step,
            seqs=multimer_object.input_seqs,
        )



def predict_multimers(multimers):
    """
    Final function to predict multimers

    Args
    multimers: A list of multimer objects created by create_multimer_objects()
    """
    for object in multimers:
        logging.info('object: '+object.description)
        path_to_models = os.path.join(FLAGS.output_path, object.description)
        logging.info(f"Modeling new interaction for {path_to_models}")
        if isinstance(object, MultimericObject):
            model_runners, random_seed = create_model_runners_and_random_seed(
                "multimer",
                1,
                FLAGS.random_seed,
                FLAGS.data_dir,
                1,
            )
            predict_individual_jobs(
                object,
                FLAGS.output_path,
                model_runners=model_runners,
                random_seed=random_seed,
            )
        else:
            model_runners, random_seed = create_model_runners_and_random_seed(
                "monomer_ptm",
                1,
                FLAGS.random_seed,
                FLAGS.data_dir,
                1,
            )
            logging.info("will run in monomer mode")
            predict_individual_jobs(
                object,
                FLAGS.output_path,
                model_runners=model_runners,
                random_seed=random_seed,
            )



def search_extension(folder_path, extension):
    files_with_extension = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                files_with_extension.append(os.path.join(root, file))
    return files_with_extension



def get_plDDT_from_result_pkl(pkl_path):
    """Get the score from the model result pkl file"""

    with open(pkl_path, "rb") as f:
        result = pickle.load(f)
    if "iptm" in result:
        score_type = "iptm+ptm"
        score = 0.8 * result["iptm"] + 0.2 * result["ptm"]
    else:
        score_type = "plddt"
        score = np.mean(result["plddt"])

    return score_type, score



def parse_atm_record(line):
    '''Get the atm record
    '''
    record = defaultdict()
    record['name'] = line[0:6].strip()
    record['atm_no'] = int(line[6:11])
    record['atm_name'] = line[12:16].strip()
    record['atm_alt'] = line[17]
    record['res_name'] = line[17:20].strip()
    record['chain'] = line[21]
    record['res_no'] = int(line[22:26])
    record['insert'] = line[26].strip()
    record['resid'] = line[22:29]
    record['x'] = float(line[30:38])
    record['y'] = float(line[38:46])
    record['z'] = float(line[46:54])
    record['occ'] = float(line[54:60])
    record['B'] = float(line[60:66])

    return record

def read_pdb(pdbfile):
    '''Read a pdb file predicted with AF and rewritten to conatin all chains
    '''

    chain_coords, chain_plddt = {}, {}
    with open(pdbfile, 'r') as file:
        for line in file:
            if not line.startswith('ATOM'):
                continue
            record = parse_atm_record(line)
            #Get CB - CA for GLY
            if record['atm_name']=='CB' or (record['atm_name']=='CA' and record['res_name']=='GLY'):
                if record['chain'] in [*chain_coords.keys()]:
                    chain_coords[record['chain']].append([record['x'],record['y'],record['z']])
                    chain_plddt[record['chain']].append(record['B'])
                else:
                    chain_coords[record['chain']] = [[record['x'],record['y'],record['z']]]
                    chain_plddt[record['chain']] = [record['B']]


    #Convert to arrays
    for chain in chain_coords:
        chain_coords[chain] = np.array(chain_coords[chain])
        chain_plddt[chain] = np.array(chain_plddt[chain])

    return chain_coords, chain_plddt


def calc_pdockq(chain_coords, chain_plddt, t):
    '''Calculate the pDockQ scores
    pdockQ = L / (1 + np.exp(-k*(x-x0)))+b
    L= 0.724 x0= 152.611 k= 0.052 and b= 0.018
    '''

    #Get coords and plddt per chain
    ch1, ch2 = [*chain_coords.keys()]
    coords1, coords2 = chain_coords[ch1], chain_coords[ch2]
    plddt1, plddt2 = chain_plddt[ch1], chain_plddt[ch2]

    #Calc 2-norm
    mat = np.append(coords1, coords2,axis=0)
    a_min_b = mat[:,np.newaxis,:] -mat[np.newaxis,:,:]
    dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
    l1 = len(coords1)
    contact_dists = dists[:l1,l1:] #upper triangular --> first dim = chain 1
    contacts = np.argwhere(contact_dists<=t)

    if contacts.shape[0]<1:
        pdockq=0
    else:
        #Get the average interface plDDT
        avg_if_plddt = np.average(np.concatenate([plddt1[np.unique(contacts[:,0])], plddt2[np.unique(contacts[:,1])]]))
        #Get the number of interface contacts
        n_if_contacts = contacts.shape[0]
        x = avg_if_plddt*np.log10(n_if_contacts)
        pdockq = 0.724 / (1 + np.exp(-0.052*(x-152.611)))+0.018

    return pdockq


def post_processing(output_dir):
    full_path = os.path.join(os.getcwd(), output_dir)
    subfolders = os.listdir(output_dir)
#Iterate over folders
    names = []
    pdockq = []
    iptm = []
    ipTM_pTM = []
    plDDT = []
    for subfolder in subfolders:
        names.append(subfolder)
        single_dir = os.path.join(full_path,subfolder)
        pkl_file = pickle.load(open(os.path.join(single_dir, f"result_{subfolder}.pkl"),'rb'))

# Obtain ipTM
        iptm_score = pkl_file['iptm']
        iptm.append(iptm_score)
        iptM_pTM_score = 0.8 * pkl_file["iptm"] + 0.2 * pkl_file["ptm"]
        ipTM_pTM.append(iptM_pTM_score)
# Mean plDDT
        meanplddt = np.mean( pkl_file["plddt"])
        plDDT.append(meanplddt)

#  Read chains
        chain_coords, chain_plddt = read_pdb(os.path.join(single_dir, f"unrelaxed_{subfolder}.pdb"))
#Check chains
        if len(chain_coords.keys())<2:
            print('Only one chain in pdbfile')
            sys.exit()

# Calculate pdockq
        t=8 #Distance threshold, set to 8 Å
        compute_pdockq = calc_pdockq(chain_coords, chain_plddt, t)
        pdockq.append(compute_pdockq)

    dict = {'name':names, 'pDockQ': pdockq, 'ipTM':iptm, 'ipTM+pTM':ipTM_pTM, 'mplDDT':plDDT}
    collect_data = pd.DataFrame(dict)
    name_result = str(FLAGS.output_path) + "_scoring.tsv"
    collect_data.to_csv(name_result, sep='\t', index=False)

       
            
# Help FLAGS

run_af = get_run_alphafold()
flags = run_af.flags

flags.DEFINE_enum(
    "mode",
    "pulldown",
    ["pulldown", "all_vs_all"],
    "Choose the mode of running multimer jobs [REQUIRED]"
)
flags.DEFINE_string(
    "output_path", None, "Folder where the data will be stored [REQUIRED]", short_name='o'
)

flags.DEFINE_list(
    "monomer_objects_dir",
    None,
    "A list of directories where monomer objects are stored [REQUIRED]", short_name='m'
)
flags.DEFINE_list("protein_list", None, "File containing a list of the names of the proteins [REQUIRED]", short_name='l')

flags.DEFINE_list("bait_list", None, "File containing a list of the names of the bait proteins (should have the same names used for the msa) [REQUIRED in pulldown mode]", short_name='b')

delattr(flags.FLAGS, "data_dir")
flags.DEFINE_string("data_dir", None, "Path to Alphafold databases directory [REQUIRED]", short_name='d')

flags.DEFINE_integer(
    "seq_index", None, "Index (number) of sequence in the fasta file to start from", short_name='n'
)
flags.DEFINE_boolean(
    "no_pair_msa", True, "do not pair the MSAs when constructing multimer objects"
)

flags.mark_flag_as_required("output_path")

delattr(flags.FLAGS, "models_to_relax")
flags.DEFINE_boolean(
    "relaxation_step",
    False,
    "Enable final relaxation step",
)


unused_flags = (
    'bfd_database_path',
    'db_preset',
    'fasta_paths',
    'hhblits_binary_path',
    'hhsearch_binary_path',
    'hmmbuild_binary_path',
    'hmmsearch_binary_path',
    'jackhmmer_binary_path',
    'kalign_binary_path',
    'max_template_date',
    'mgnify_database_path',
    'num_multimer_predictions_per_model',
    'obsolete_pdbs_path',
    'output_dir',
    'pdb70_database_path',
    'pdb_seqres_database_path',
    'small_bfd_database_path',
    'template_mmcif_dir',
    'uniprot_database_path',
    'uniref30_database_path',
    'uniref90_database_path',
)

for flag in unused_flags:
    delattr(flags.FLAGS, flag)

FLAGS = flags.FLAGS



# Main 
def main(argv):
    check_output_dir(FLAGS.output_path)
    
    if FLAGS.mode == "pulldown":
        bait_proteins = read_all_proteins(FLAGS.bait_list[0])
        candidate_proteins = []
        for file in FLAGS.protein_list:
            candidate_proteins.append(read_all_proteins(file))
        data = create_pulldown_info(
            bait_proteins, candidate_proteins, seq_index=FLAGS.seq_index
        )
        multimers = create_multimer_objects(data, FLAGS.monomer_objects_dir, not FLAGS.no_pair_msa)


    elif FLAGS.mode == "all_vs_all":
        all_proteins = read_all_proteins(FLAGS.protein_list)
        data = create_all_vs_all_info(all_proteins, seq_index=FLAGS.seq_index)
        multimers = create_multimer_objects(data, FLAGS.monomer_objects_dir, not FLAGS.no_pair_msa)


    predict_multimers(multimers)
    post_processing(FLAGS.output_path)


if __name__ == "__main__":
    app.run(main)
