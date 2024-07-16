#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 10:58:36 2022

@author: zhiyuas
"""

import Bio.PDB
import numpy as np
structure = []
seq_str = 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA'
#for i in range(2):
#    structure.append(Bio.PDB.PDBParser().get_structure('{}'.format(i),'movies/{}.pdb'.format(i)))
structure = Bio.PDB.PDBParser().get_structure('all','./movies/all.pdb')
ref_model = structure[0]
for alt_model in structure :
    #Build paired lists of c-alpha atoms:
    ref_atoms = []
    alt_atoms = []
    for (ref_chain, alt_chain) in zip(ref_model, alt_model) :
        for ref_res, alt_res in zip(ref_chain, alt_chain) :
            '''
            #CA = alpha carbon
            ref_atoms.append(ref_res['CA'])             
            alt_atoms.append(alt_res['CA'])
            '''
            #bb atoms
            for bb_atm in ['N','C','CA','O']:
                ref_atoms.append(ref_res[bb_atm])
                alt_atoms.append(alt_res[bb_atm])
    #Align these paired atom lists:
    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(ref_atoms, alt_atoms)
    if ref_model.id == alt_model.id :
        #Check for self/self get zero RMS, zero translation
        #and identity matrix for the rotation.
        assert np.abs(super_imposer.rms) < 0.0000001
        assert np.max(np.abs(super_imposer.rotran[1])) < 0.000001
        assert np.max(np.abs(super_imposer.rotran[0]) - np.identity(3)) < 0.000001
    else :
        #Update the structure by moving all the atoms in
        #this model (not just the ones used for the alignment)
        super_imposer.apply(alt_model.get_atoms())
    #print ("RMS(first model, model %i) = %0.2f" % (alt_model.id, super_imposer.rms))
'''    
data = []
for model in structure:
    atm_coord = []
    for chain in model:
        for resi in chain:
            for atm in resi:
                atm_coord.extend(atm.coord)
    data.append(atm_coord)
np.savetxt('project/all_atm_coords.txt',np.array(data))
'''
io=Bio.PDB.PDBIO()
io.set_structure(structure)
io.save('all_aligned.pdb')
