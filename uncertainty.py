import numpy as np
from ase.db import connect
from tqdm import tqdm
from jsex.nnap import NNAP

def select_by_energy_uncertainty(db_in_path: str, db_out_path: str, top_k: int, jnn_path: str):
    """
    Selects structures based on the energy uncertainty criterion defined in the manuscript.
    
    This function computes the per-atom energy prediction error U(S) between the 
    DFT-computed energy and the NNAP-predicted energy. It ranks all candidate structures 
    in descending order of their uncertainty and extracts the top-k structures to form 
    the error set S_err for subsequent training.
    
    Args:
        db_in_path (str): Path to the input ASE database.
        db_out_path (str): Path to save the extracted high-uncertainty structures.
        top_k (int): Number of top structures with the highest uncertainty to select (k parameter).
        dft_energy_
    """

    print(f"[*] Reading structures from {db_in_path} to evaluate Energy Uncertainty...")

    uncertainty_data = []

    with connect(db_in_path) as db:
        rows = list(db.select())
        total_structures = len(rows)

    if top_k >= total_structures:
        print(f"[!] Warning: Target selection size ({top_k}) is greater than or equal to the pool size ({total_structures}).")
        top_k = total_structures

    calc = NNAP(jnn_path).asAseCalculator()

    for row in tqdm(rows, desc="Evaluating U(S)", unit="struct"):
        atoms = row.toatoms()
        atoms.calc = calc
        N_atom = len(atoms)
        E_NNAP = atoms.get_potential_energy()

        try:
            E_DFT = row.energy 
            if E_DFT is None:
                raise ValueError("The DB row does not contain a primary energy value.")
        except Exception as e:
            raise KeyError(f"Failed to read DFT energy from DB: {e}. Make sure the DFT energy is stored properly.")
        
        U_S = abs(E_DFT - E_NNAP) / N_atom
        uncertainty_data.append((row.id, atoms, U_S))

    print("[*] Ranking structures by uncertainty...")
    uncertainty_data.sort(key=lambda x: x[2], reverse=True)

    selected_structures = uncertainty_data[:top_k]
    U_th = selected_structures[-1][2]
    max_U = selected_structures[0][2]

    print(f"  -> Maximum Uncertainty found:  {max_U:.6f} eV/atom")
    print(f"  -> Threshold Uncertainty U_th: {U_th:.6f} eV/atom (top {top_k} structures)")
    print(f"[*] Saving the top {top_k} high-uncertainty configurations to {db_out_path}...")

    with connect(db_out_path) as db_out:
        for idx, atoms, u_val in tqdm(selected_structures, desc="Saving S_err", unit="struct"):
            atoms.calc = None
            db_out.write(atoms, key_value_pairs={'U_S': u_val, 'selection_criterion': 'high_uncertainty'})

    print("[*] Done! Uncertainty-based selection completed.")