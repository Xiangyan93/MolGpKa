#!/usr/bin/env python
# coding: utf-8
from typing import Dict
import os
from .predict_pka import *


class MolGpKa:
    def __init__(self) -> None:
        # get the location of current file
        CWD = os.path.dirname(os.path.abspath(__file__))
        self.model_acid = load_model(
            os.path.join(CWD, "models/weight_acid.pth"))
        self.model_base = load_model(
            os.path.join(CWD, "models/weight_base.pth"))

    def predict(self, mol, uncharged=True):
        if uncharged:
            un = rdMolStandardize.Uncharger()
            mol = un.uncharge(mol)
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        mol = AllChem.AddHs(mol)
        base_dict = self._predict_base(mol)
        acid_dict = self._predict_acid(mol)
        return base_dict, acid_dict, mol

    def CalculateMolCharge(self, mol, pH: float = 7.0):
        mol_charge = 0.
        base_dict, acid_dict, mol = self.predict(mol)
        # calculate the format charge at specific pH
        for atom_id, pka in base_dict.items():
            atom = mol.GetAtomWithIdx(atom_id)
            assert atom.GetFormalCharge() == 0
            x = 10 ** (pka - pH)
            mol_charge += x / (x + 1)
        for atom_id, pka in acid_dict.items():
            atom = mol.GetAtomWithIdx(atom_id)
            assert atom.GetFormalCharge() == 0
            x = 10 ** (pH - pka)
            mol_charge -= x / (x + 1)
        return mol_charge

    def GetSmilesInSolution(self, mol, pH: float = 7.0):
        base_dict, acid_dict, mol = self.predict(mol)
        mol = Chem.RWMol(mol)
        for atom_id, pka in base_dict.items():
            atom = mol.GetAtomWithIdx(atom_id)
            assert atom.GetFormalCharge() == 0
            x = 10 ** (pka - pH)
            if x / (x + 1) > 0.5:
                mol.GetAtomWithIdx(atom_id).SetFormalCharge(1)
                hydrogen = Chem.Atom(1)
                mol.AddAtom(hydrogen)
                bond = Chem.BondType.SINGLE
                mol.AddBond(atom_id, mol.GetNumAtoms() - 1, bond)
                
        removed_atoms = []
        for atom_id, pka in acid_dict.items():
            atom = mol.GetAtomWithIdx(atom_id)
            assert atom.GetFormalCharge() == 0
            assert atom.GetAtomicNum() == 1
            assert len(atom.GetNeighbors()) == 1
            x = 10 ** (pH - pka)
            if x / (x + 1) > 0.5:
                mol.GetAtomWithIdx(atom_id).GetNeighbors()[0].SetFormalCharge(-1)
                removed_atoms.append(atom_id)
        for atom_id in sorted(removed_atoms, reverse=True):
            mol.RemoveAtom(atom_id)
        mol = AllChem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)

    def _predict_base(self, mol) -> Dict[int, float]:
        base_idxs = get_ionization_aid(mol, acid_or_base="base")
        base_res = {}
        for aid in base_idxs:
            bpka = model_pred(mol, aid, self.model_base)
            base_res.update({aid: bpka})
        return base_res

    def _predict_acid(self, mol) -> Dict[int, float]:
        acid_idxs = get_ionization_aid(mol, acid_or_base="acid")
        acid_res = {}
        for aid in acid_idxs:
            apka = model_pred(mol, aid, self.model_acid)
            acid_res.update({aid: apka})
        return acid_res
