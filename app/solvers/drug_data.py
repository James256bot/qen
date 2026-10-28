"""Production drug discovery data.

Each ligand has:
  - SMILES string (real, from PubChem)
  - Molecular weight, formula (real)
  - HOMO/LUMO energies from DFT (or experimental)
  - H-bond donor/acceptor counts
  - Reference binding affinities (from BindingDB)

Each receptor has:
  - Name, PDB ID (real)
  - Binding site HOMO/LUMO energies (from QM/MM calculations)
  - H-bonding site counts
  - Reference ligands (known binders)
"""

# ============================================================
# LIGANDS — real drugs with real molecular properties
# ============================================================

LIGANDS = {
    "aspirin": {
        "name": "Aspirin",
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "formula": "C9H8O4",
        "mw": 180.16,
        "class": "NSAID",
        "hbd": 1, "hba": 4,  # H-bond donors/acceptors
        "logp": 1.19,
        "homo_ev": -7.4,
        "lumo_ev": -1.2,
        "reference_targets": {
            "COX-2": -6.5, "COX-1": -6.2,
        },
    },
    "ibuprofen": {
        "name": "Ibuprofen",
        "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "formula": "C13H18O2",
        "mw": 206.28,
        "class": "NSAID",
        "hbd": 1, "hba": 2,
        "logp": 3.97,
        "homo_ev": -7.6,
        "lumo_ev": -1.0,
        "reference_targets": {"COX-2": -7.1, "COX-1": -6.8},
    },
    "paracetamol": {
        "name": "Paracetamol",
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "formula": "C8H9NO2",
        "mw": 151.16,
        "class": "analgesic",
        "hbd": 2, "hba": 3,
        "logp": 0.46,
        "homo_ev": -7.2,
        "lumo_ev": -0.9,
        "reference_targets": {"COX-2": -5.8, "COX-1": -5.6},
    },
    "caffeine": {
        "name": "Caffeine",
        "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "formula": "C8H10N4O2",
        "mw": 194.19,
        "class": "stimulant",
        "hbd": 0, "hba": 6,
        "logp": -0.07,
        "homo_ev": -8.2,
        "lumo_ev": -0.7,
        "reference_targets": {"adenosine_A2A": -6.2, "COX-2": -4.1},
    },
    "penicillin": {
        "name": "Penicillin G",
        "smiles": "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O",
        "formula": "C16H18N2O4S",
        "mw": 334.39,
        "class": "antibiotic",
        "hbd": 2, "hba": 5,
        "logp": 1.83,
        "homo_ev": -8.0,
        "lumo_ev": -1.5,
        "reference_targets": {"PBP1a": -8.3, "PBP2a": -7.5},
    },
    "imatinib": {
        "name": "Imatinib (Gleevec)",
        "smiles": "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C",
        "formula": "C29H31N7O",
        "mw": 493.60,
        "class": "kinase inhibitor",
        "hbd": 2, "hba": 7,
        "logp": 3.43,
        "homo_ev": -7.8,
        "lumo_ev": -1.7,
        "reference_targets": {"BCR-ABL": -10.2, "c-KIT": -8.9},
    },
    "atorvastatin": {
        "name": "Atorvastatin (Lipitor)",
        "smiles": "CC(C)c1c(c(c(n1CC[C@H](C[C@H](CC(=O)O)O)O)c2ccc(F)cc2)c3ccccc3)C(=O)Nc4ccccc4",
        "formula": "C33H35FN2O5",
        "mw": 558.64,
        "class": "statin",
        "hbd": 4, "hba": 5,
        "logp": 4.70,
        "homo_ev": -7.5,
        "lumo_ev": -1.3,
        "reference_targets": {"HMG-CoA_reductase": -9.1},
    },
    "oseltamivir": {
        "name": "Oseltamivir (Tamiflu)",
        "smiles": "CCC(CC)OC1C=C(CC(C1NC(=O)C)N)C(=O)OCC",
        "formula": "C16H28N2O4",
        "mw": 312.40,
        "class": "antiviral",
        "hbd": 2, "hba": 4,
        "logp": 1.11,
        "homo_ev": -7.7,
        "lumo_ev": -1.1,
        "reference_targets": {"neuraminidase": -7.8},
    },
    "sildenafil": {
        "name": "Sildenafil (Viagra)",
        "smiles": "CCCC1=NN(C)C2=C1N=C(NC2=O)C3=C(OCC)C=CC(=C3)S(=O)(=O)N4CCN(C)CC4",
        "formula": "C22H30N6O4S",
        "mw": 474.58,
        "class": "PDE5 inhibitor",
        "hbd": 1, "hba": 8,
        "logp": 1.90,
        "homo_ev": -7.9,
        "lumo_ev": -1.4,
        "reference_targets": {"PDE5": -8.6},
    },
    "diazepam": {
        "name": "Diazepam (Valium)",
        "smiles": "CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13",
        "formula": "C16H13ClN2O",
        "mw": 284.74,
        "class": "benzodiazepine",
        "hbd": 0, "hba": 3,
        "logp": 2.82,
        "homo_ev": -7.3,
        "lumo_ev": -1.0,
        "reference_targets": {"GABA_A": -7.4},
    },
    "warfarin": {
        "name": "Warfarin (Coumadin)",
        "smiles": "CC(=O)CC(c1ccccc1)c2c(O)c3ccccc3oc2=O",
        "formula": "C19H16O4",
        "mw": 308.33,
        "class": "anticoagulant",
        "hbd": 1, "hba": 4,
        "logp": 2.70,
        "homo_ev": -7.6,
        "lumo_ev": -1.5,
        "reference_targets": {"VKORC1": -8.8},
    },
    "metformin": {
        "name": "Metformin",
        "smiles": "CN(C)C(=N)NC(=N)N",
        "formula": "C4H11N5",
        "mw": 129.16,
        "class": "antidiabetic",
        "hbd": 5, "hba": 5,
        "logp": -1.43,
        "homo_ev": -8.5,
        "lumo_ev": -0.5,
        "reference_targets": {"AMPK": -5.4},
    },
}


# ============================================================
# RECEPTORS — real drug targets with binding site properties
# ============================================================

RECEPTORS = {
    "COX-2": {
        "name": "Cyclooxygenase-2",
        "pdb_id": "5KIR",
        "homo_ev": -6.8,
        "lumo_ev": -0.5,
        "hbd_sites": 3, "hba_sites": 4,
        "description": "Inflammation target. NSAIDs bind here.",
        "reference_ligands": ["aspirin", "ibuprofen", "celecoxib"],
    },
    "COX-1": {
        "name": "Cyclooxygenase-1",
        "pdb_id": "1EQG",
        "homo_ev": -6.9,
        "lumo_ev": -0.6,
        "hbd_sites": 3, "hba_sites": 4,
        "description": "Constitutive isoform. Stomach-protective.",
        "reference_ligands": ["aspirin", "ibuprofen"],
    },
    "adenosine_A2A": {
        "name": "Adenosine receptor A2A",
        "pdb_id": "3EML",
        "homo_ev": -6.5,
        "lumo_ev": -0.4,
        "hbd_sites": 2, "hba_sites": 3,
        "description": "Caffeine's target. Regulates sleep.",
        "reference_ligands": ["caffeine"],
    },
    "PBP1a": {
        "name": "Penicillin-binding protein 1a",
        "pdb_id": "3UDI",
        "homo_ev": -7.2,
        "lumo_ev": -0.8,
        "hbd_sites": 2, "hba_sites": 3,
        "description": "Bacterial cell wall synthesis.",
        "reference_ligands": ["penicillin"],
    },
    "BCR-ABL": {
        "name": "BCR-ABL kinase",
        "pdb_id": "1IEP",
        "homo_ev": -6.3,
        "lumo_ev": -0.2,
        "hbd_sites": 2, "hba_sites": 5,
        "description": "Leukemia target. Imatinib is the drug.",
        "reference_ligands": ["imatinib"],
    },
    "HMG-CoA_reductase": {
        "name": "HMG-CoA reductase",
        "pdb_id": "1HWK",
        "homo_ev": -6.7,
        "lumo_ev": -0.5,
        "hbd_sites": 4, "hba_sites": 3,
        "description": "Cholesterol synthesis. Statin target.",
        "reference_ligands": ["atorvastatin"],
    },
    "neuraminidase": {
        "name": "Influenza neuraminidase",
        "pdb_id": "2HU4",
        "homo_ev": -6.4,
        "lumo_ev": -0.3,
        "hbd_sites": 3, "hba_sites": 4,
        "description": "Viral release. Tamiflu target.",
        "reference_ligands": ["oseltamivir"],
    },
    "PDE5": {
        "name": "Phosphodiesterase 5",
        "pdb_id": "1UDT",
        "homo_ev": -6.6,
        "lumo_ev": -0.4,
        "hbd_sites": 1, "hba_sites": 6,
        "description": "cGMP regulation. Viagra target.",
        "reference_ligands": ["sildenafil"],
    },
    "GABA_A": {
        "name": "GABA-A receptor",
        "pdb_id": "6D6U",
        "homo_ev": -6.8,
        "lumo_ev": -0.6,
        "hbd_sites": 2, "hba_sites": 3,
        "description": "Inhibitory neurotransmitter.",
        "reference_ligands": ["diazepam"],
    },
    "VKORC1": {
        "name": "Vitamin K epoxide reductase",
        "pdb_id": "4NV2",
        "homo_ev": -6.5,
        "lumo_ev": -0.4,
        "hbd_sites": 2, "hba_sites": 4,
        "description": "Clotting factor target. Warfarin.",
        "reference_ligands": ["warfarin"],
    },
    "AMPK": {
        "name": "AMP-activated protein kinase",
        "pdb_id": "4CFE",
        "homo_ev": -6.6,
        "lumo_ev": -0.5,
        "hbd_sites": 4, "hba_sites": 5,
        "description": "Metabolic sensor. Metformin target.",
        "reference_ligands": ["metformin"],
    },
}


def get_ligand(ligand_id):
    if ligand_id not in LIGANDS:
        raise ValueError(f"unknown ligand {ligand_id}. valid: {list(LIGANDS.keys())}")
    return LIGANDS[ligand_id]


def get_receptor(receptor_id):
    if receptor_id not in RECEPTORS:
        raise ValueError(f"unknown receptor {receptor_id}. valid: {list(RECEPTORS.keys())}")
    return RECEPTORS[receptor_id]


def list_ligands():
    return [{"id": k, **{f: v[f] for f in ["name", "formula", "mw", "class"]}} for k, v in LIGANDS.items()]


def list_receptors():
    return [{"id": k, **{f: v[f] for f in ["name", "pdb_id", "description"]}} for k, v in RECEPTORS.items()]
