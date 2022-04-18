import sys
import os
import subprocess

# TODO: Machine Learning (ML) was removed; 
# re-train the original data from 
# https://github.com/ShirleyWISiu/LigTMap
#from sklearn.ensemble import RandomForestRegressor

# TODO: When the new models are available, uncomment 
# and finish model persistence implementations
"""
import pickle
import joblib
import sklearn_json as skljson      # Not supported yet!
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from multiprocessing import cpu_count
"""

import pandas as pd
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from rdkit.Chem.MolStandardize import rdMolStandardize
from oddt import fingerprints, toolkit

smile_inf_name = sys.argv[1]
input_num = sys.argv[2]
select_db = sys.argv[3]
total_db = int(sys.argv[4])
rootpa = sys.argv[5]
output_path = f'Output/Input_{input_num}/{select_db}'
summary_path = f'Output/Input_{input_num}'
model_fmt = 'joblib'
unsupported_fmts = ['sklearn-json', 'onnx', 'pmml']

try:
    model_fmt = sys.argv[6]
    if not model_fmt:
        raise Exception('Please specify one of the model (persisted) formats: pickle, joblib, sklearn-json, onnx, pmml')
except IndexError:
    raise Exception('Please specify one of the model (persisted) formats: pickle, joblib, sklearn-json, onnx, pmml')

if model_fmt in unsupported_fmts:
    raise Exception(f'{model_fmt} not supported yet!')


def __override_machine__(test, y):
    cpus = cpu_count()
    #model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, #  FutureWarning: Criterion 'mse' was deprecated in v1.0 and will be removed in version 1.2. Use `criterion='squared_error'` which is equivalent
    model = RandomForestRegressor(bootstrap=True, criterion='squared_error', max_depth=None,
        max_features='auto', max_leaf_nodes=None,
        # min_impurity_split=None,    # Deprecated since version 0.19: min_impurity_split has been deprecated in favor of min_impurity_decrease in 0.19 and (will be) removed in 0.21. Use min_impurity_decrease instead.
        min_impurity_decrease=0.0,
        min_samples_leaf=1, min_samples_split=2,
        min_weight_fraction_leaf=0.0, n_estimators=10, 
        n_jobs = cpus,
        oob_score=False, random_state=None, verbose=0, warm_start=False)

    try:
        predictions = model.predict(test)
    except Exception as e:
        if 'is not fitted yet' in e.args[0]:
            model.fit(test, y)
            predictions = model.predict(test)
            score = model.score(test, y)

    #predictions = [np.round(x) for x in predictions]
    return model, predictions, score

smile_inf = open(smile_inf_name, 'r')
smile = smile_inf.readline().rstrip()
smile_inf.close()
tanifingcut = 0.4
output_list = []

try:        
    status_f = open('Output/status.txt', 'a')

    print("Step 1: Ligand Similarity Search")
    #### PART1: Similarity <--> Tanifing + Tanipharm Score ####

    try:
        # Generate the fingerprint
        #mol1 = Chem.MolFromSmiles(smile)
        """
        To avoid: "Explicit valence for atom is greater than permitted"
        """
        standarized_smile = rdMolStandardize.StandardizeSmiles(smile)
        mol1 = Chem.MolFromSmiles(standarized_smile)
        # Ends avoid "Explicit valence for atom is greater than permitted"

        """
        To avoid: "Molecule does not have explicit Hs. Consider calling AddHs()"
        """
        mol1 = Chem.AddHs(mol1)
        # Ends avoid "Molecule does not have explicit Hs. Consider calling AddHs()"

        AllChem.EmbedMolecule(mol1, useRandomCoords=True)
        FP_inp_Morgan = AllChem.GetMorganFingerprint(mol1, radius=2)
        FP_inp_MACCSF = AllChem.GetMACCSKeysFingerprint(mol1)
        FP_inp_Daylight = AllChem.RDKFingerprint(mol1)
    except BaseException:
        smile_err_file = open(f'{summary_path}/smile_err.dat', 'w')
        smile_err_file.close()

    id_ligname = {}
    id_proname = {}

    index_file = open(f'{rootpa}/Index/{select_db}.csv', 'r')
    label_line = index_file.readline()
    for y in index_file:
        try:
            y = y.rstrip()
            data_pdb = y.split(';')[0]
            data_smile = y.split(';')[1]
            affinity = y.split(';')[2]
            id_proname[data_pdb] = y.split(';')[3]
            id_ligname[data_pdb] = y.split(';')[4]

            ################ Fing part ###############
            # ECFP2 Morgan
            try:
                data_morgan_file = open(f'{rootpa}/Index/fing/Morgan/{select_db}/{data_pdb}.bin', 'rb')
                data_morgan_bin = data_morgan_file.readline().rstrip()
                data_morgan = DataStructs.UIntSparseIntVect(data_morgan_bin)
            except Exception as e2:
                if 'failed to read from stream' in e2.args[0]:
                    try:
                        data_morgan_file = open(f'{rootpa}/Index/fing/Morgan/{select_db}/{data_pdb}.new.bin', 'rb')
                        data_morgan_bin = data_morgan_file.readline().rstrip()
                        data_morgan = DataStructs.UIntSparseIntVect(data_morgan_bin)
                    except Exception as e3:
                        if len(e3.args) > 0 and 'No such file or directory' in e3.args[1]:
                            try:
                                mol2 = Chem.MolFromPDBFile(f'{rootpa}protein/{select_db}/{data_pdb}.pdb')
                            except OSError as e:
                                cmd = [
                                    'bash', 'pdb-download', 
                                    f'{rootpa}protein/{select_db}/', data_pdb
                                ]
                                subprocess.Popen(cmd, stdout=subprocess.PIPE)
                                mol2 = Chem.MolFromPDBFile(f'{rootpa}/protein/{select_db}/{data_pdb}.pdb')

                            data_morgan = AllChem.GetMorganFingerprint(mol2, radius=2)
                            new_finger = open(f'{rootpa}/Index/fing/Morgan/{select_db}/{data_pdb}.new.bin', 'wb')
                            new_finger.write(data_morgan.ToBinary())
                            new_finger.close()

            # MACCSF
            data_maccsf_file = open(f'{rootpa}/Index/fing/MACCSF/New/{select_db}/{data_pdb}.dat', 'r')
            data_maccsf_bin = data_maccsf_file.readline().rstrip()
            data_maccsf = DataStructs.CreateFromBitString(data_maccsf_bin)

            ############# Calculate the Tani Score ###########
            Tanifing_Morgan = DataStructs.TanimotoSimilarity(FP_inp_Morgan, data_morgan)
            Tanifing_MACCSF = DataStructs.TanimotoSimilarity(FP_inp_MACCSF, data_maccsf)

            if total_db > 1:
                # Daylight part
                data_daylight_file = open(f'{rootpa}/Index/fing/Daylight/{select_db}/{data_pdb}.dat', 'r')
                data_daylight_bin = data_daylight_file.readline().rstrip()
                data_daylight = DataStructs.CreateFromBitString(data_daylight_bin)
                Tanifing_Daylight = DataStructs.TanimotoSimilarity(FP_inp_Daylight, data_daylight)
                sum_score = (Tanifing_Morgan + Tanifing_MACCSF + Tanifing_Daylight) / 3
            else:
                sum_score = (Tanifing_Morgan + Tanifing_MACCSF) / 2

            if sum_score >= tanifingcut:
                row = [
                    smile,
                    data_smile,
                    data_pdb,
                    affinity,
                    str(Tanifing_Morgan),
                    str(Tanifing_MACCSF),
                    str(sum_score)]
                output_list.append(row)
        except BaseException as e:
            #print('*********** BaseException *********\n')
            #print(e)
            error_file = open(f'{output_path}/error.txt', 'a')
            error_file.write(f'{data_smile}\n')
            error_file.close()
            continue

    #### PART1 COMPLETED ####

    print("Step 2: Docking")
    #### PART2: Docking ####

    # Prepare the input ligand
    """
    command = "obabel -:" + smile + " -opdb -O " + \
        output_path + "/input.pdb --gen3d "
    """
    command = f'obabel -:{smile} -opdb -O {output_path}/input.pdb --gen3d '
    subprocess.check_output(command.split())

    ####################
    """
    Fixes:
        raise AssertionError , "%s does't exist" %filename
        AssertionError: input.pdb does't exist
    """
    ####################
    from pathlib import Path
    folder = Path(__file__).resolve().parent.joinpath(output_path)
    os.system('cp ' + str(folder) + '/input.pdb .')
    ####################
    # Ends
    ####################

    """
    command = os.environ.get('MGLTools') + "/bin/pythonsh " + os.environ.get('MGLTools') + "/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py -l " + \
        output_path + "/input.pdb -o " + output_path + \
        "/input.pdbqt -A 'hydrogens' -U 'nphs_lps_waters'"
    """
    mgl_tools = os.environ.get('MGLTools')
    command = [
        f'{mgl_tools}/bin/pythonsh', 
        f'{mgl_tools}/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py -l', 
        f'{output_path}/input.pdb -o', 
        f"{output_path}/input.pdbqt -A 'hydrogens' -U 'nphs_lps_waters'"
    ]
    subprocess.check_output(command)

    # Get the PDBID
    out_file = open(f'{output_path}/file_list', 'w')

    pid_list = []
    for ele in output_list:
        out_file.write(f'{ele[2]}\n')
        pid_list.append(ele[2])

    out_file.close()

    # Docking
    command = [
        f'{rootpa}/docking_files/dock.sh', 'pso',
        f'{output_path}', '1', '1', f'{rootpa}'
    ]
    subprocess.check_call(command)

    # Write the docking score to the output.csv
    score_file = open(f'{output_path}/DOCK_LOG/score_1.dat', 'r')

    score_list = score_file.readlines()
    score_file.close()

    out_file = open(f'{output_path}/output.csv', 'w')
    out_file.write(
        'Input;Data;PDB;Affinity;Tani_Morgan;Tani_MACCSF;LigandScore;ILDScore\n')

    i = 0
    for line in output_list:
        out_line = ';'.join(line)
        idscore = score_list[i].rstrip()
        if idscore != '':
            out_line = f'{out_line};{idscore}\n'
        else:
            out_line = f'{out_line};n.a.\n'
        out_file.write(out_line)
        i = i + 1

    out_file.close()

    # Create complex file
    for pdbid in pid_list:
        ligand_filename = \
            f'{output_path}/DOCK_LOG/{pdbid}/{pdbid}_ligand_1.pdb'

        protein_filename = \
            f'{rootpa}/docking_files/protein/{pdbid}_protein.pdb'

        filenames = [protein_filename, ligand_filename]
        output_filename = \
            f'{output_path}/Complex/complex_{pdbid}.pdb'

        outfile = open(output_filename, 'w')
        for fname in filenames:
            if os.path.isfile(fname):
                infile = open(fname, 'r')
                outfile.write(infile.read())

    #### PART2 COMPLETED ####

    print("Step 3: Activity Prediction")
    #### PART3: Activity prediction ####
    # Read the data, affinity prediction
    csv_name = f'{output_path}/output.csv'
    data = pd.read_csv(csv_name, sep=';', header=[0])

    # Add some new columns
    data['Mol'] = data['Input'].apply(Chem.MolFromSmiles)

    def MorganFingerprint(mol):
        return FingerprintMol(mol)[0]

    # Scale X to unit variance and zero mean
    data['Fingerprint'] = data['Mol'].apply(MorganFingerprint)

    Test = np.array(list(data['Fingerprint']))
    y = np.array(list(data['Fingerprint']), dtype=np.float32)
    score = 0.0

    # Load the model
    # TODO: When the new models are available, uncomment 
    # and finish model persistence implementations
    """
    try:
        if model_fmt == 'pickle':
            f = open(f'{rootpa}/model/{select_db}.pickle', 'rb')
            model = pickle.load(f)
        elif model_fmt == 'joblib':
            model = joblib.load(f'{rootpa}/model/{select_db}.joblib')
        elif model_fmt == 'sklearn-json':
            skljson.from_json(f'{rootpa}/model/{select_db}.skljson')
        elif model_fmt == 'onnx':
            raise Exception(f'{model_fmt} not supported yet!')
        elif model_fmt == 'pmml':
            raise Exception(f'{model_fmt} not supported yet!')
            #f = open(f'{rootpa}/model/{select_db}.pmml', 'rb')

        predictions = model.predict(Test)
        score = model.score(Test, y)
    except Exception as e:
        if len(e.args) > 0 and 'No such file or directory' in e.args[1]:
            model, predictions, score = __override_machine__(Test, y)
            if model_fmt == 'pickle':
                with open(f'{rootpa}/model/{select_db}.pickle', 'wb') as f:
                    pickle.dump(model, f)
            elif model_fmt == 'joblib':
                joblib.dump(model, f'{rootpa}/model/{select_db}.joblib')
            elif model_fmt == 'sklearn-json':
                skljson.to_json(model, f'{rootpa}/model/{select_db}.skljson')
            elif model_fmt == 'onnx':
                raise Exception(f'{model_fmt} not supported yet!')
            elif model_fmt == 'pmml':
                raise Exception(f'{model_fmt} not supported yet!')
                #try:
                #    pipeline = PMMLPipeline([("classifier", model)])
                #    sklearn2pmml(pipeline, f'{rootpa}/model/{select_db}.pmml')
                #except Exception as e:
                #    print(e)
    """

    #### PART3 COMPLETED ####

    countline = -1
    countlinefile = open(f'{output_path}/output.csv', 'r')
    for line in countlinefile:
        countline += 1

    print("Step 4: Binding Similarity Search (Total " + str(
        countline) + " target pdb, please wait...)")
    #### PART4: Similarity <--> Binding fingerprint ####

    def tanimoto(a, b, sparse=False):
        if sparse:
            a = np.unique(a)
            b = np.unique(b)
            a_b = float(len(np.intersect1d(a, b, assume_unique=True)))
            denominator = len(a) + len(b) - a_b
            if denominator > 0:
                return a_b / denominator
        else:
            a = a.astype(bool)
            b = b.astype(bool)
            a_b = (a & b).sum().astype(float)
            denominator = a.sum() + b.sum() - a_b
            if denominator > 0:
                return a_b / denominator
        return 0.

    first_result_table = open(output_path + '/output.csv', 'r')
    ignore_label = first_result_table.readline()

    for line in first_result_table:
        line_list = line.rstrip().split(';')
        r_pdbid = line_list[2]
        r_ligandscore = line_list[6]
        r_dockscore = line_list[7]

        dummy1 = os.path.isfile(f'{rootpa}/Index/IFP/{select_db}/{r_pdbid}.bin')
        dummy2 = os.path.isfile(
            f'{output_path}/DOCK_LOG/{r_pdbid}/{r_pdbid}_ligand_1.pdb')

        bindscore = 0
        ligtmapscore = 0

        if (dummy1 and dummy2):
            crystal_IFP = np.fromfile(
                f'{rootpa}/Index/IFP/{select_db}/{r_pdbid}.bin', 
                dtype=np.uint8)

            dummy1 = toolkit.readfile(
                'pdb', f'{rootpa}/docking_files/protein/{r_pdbid}_protein.pdb')
            bind_protein = next(dummy1)

            bind_protein.protein = True

            dummy1 = toolkit.readfile(
                'pdb', f'{output_path}/DOCK_LOG/{r_pdbid}/{r_pdbid}_ligand_1.pdb')
            bind_ligand = next(dummy1)

            IFP = fingerprints.InteractionFingerprint(
                bind_ligand, bind_protein)
            bindscore = tanimoto(crystal_IFP, IFP)

            ligtmapscore = 0.7 * float(r_ligandscore) + 0.3 * bindscore

            # Write the binding fingerprint score to the IFP_result.csv
            IFP_filename = f'{summary_path}/IFP_result.csv'

            if os.path.isfile(IFP_filename) != True:
                IFP_file = open(IFP_filename, 'w')
                # TODO: When the new models are available, the value for 
                # PredictedAffinity would can be added
                """
                content = \
                    'PDB;Class;TargetName;LigandName;' + \
                    'LigandSimilarityScore;' + \
                    'BindingSimilarityScore;LigTMapScore;' + \
                    'PredictedAffinity;DockingScore\n'
                """
                content = \
                    'PDB;Class;TargetName;LigandName;' + \
                    'LigandSimilarityScore;' + \
                    'BindingSimilarityScore;LigTMapScore;' + \
                    'DockingScore\n'
                IFP_file.write(content)

                # TODO: When the new models are available, the value for 
                # score would can be added
                """
                content = \
                    f'{r_pdbid};{select_db};{id_proname[r_pdbid]};' + \
                    f'{id_ligname[r_pdbid]};' + \
                    str(round(float(r_ligandscore), 6)) + ';' + \
                    str(round(bindscore, 6)) + ';' + \
                    str(round(ligtmapscore, 6)) + ';' + \
                    str(round(score, 6)) + ';' + \
                    str(round(float(r_dockscore), 6)) + '\n'
                """
                content = \
                    f'{r_pdbid};{select_db};{id_proname[r_pdbid]};' + \
                    f'{id_ligname[r_pdbid]};' + \
                    str(round(float(r_ligandscore), 6)) + ';' + \
                    str(round(bindscore, 6)) + ';' + \
                    str(round(ligtmapscore, 6)) + ';' + \
                    str(round(float(r_dockscore), 6)) + '\n'
                IFP_file.write(content)
                IFP_file.close()
            else:
                tmp_list = []
                tmp_file = open(IFP_filename, 'r')
                label_line = tmp_file.readline()
                for line in tmp_file.readlines():
                    line = line.rstrip()
                    line = line.split(';')
                    tmptmp_list = []
                    #for i in range(9):
                    for i in range(8):                  # score (from scikit learn predict) was removed
                        tmptmp_list.append(line[i])
                    tmp_list.append(tmptmp_list)
                tmp_file.close()

                # TODO: When the new models are available, the value for 
                # score would can be added
                """
                data = [
                    r_pdbid, select_db, id_proname[r_pdbid], id_ligname[r_pdbid],
                    str(round(float(r_ligandscore), 6)),
                    str(round(bindscore, 6)),
                    str(round(ligtmapscore, 6)),
                    str(round(score, 6)),
                    str(round(float(r_dockscore), 6))
                ]
                """
                data = [
                    r_pdbid, select_db, id_proname[r_pdbid], id_ligname[r_pdbid],
                    str(round(float(r_ligandscore), 6)),
                    str(round(bindscore, 6)),
                    str(round(ligtmapscore, 6)),
                    str(round(float(r_dockscore), 6))
                ]
                tmp_list.append(data)
                tmp_list.sort(key=lambda s: s[6], reverse=True)
                IFP_file = open(IFP_filename, 'w')
                IFP_file.write(label_line)
                for ele in tmp_list:
                    write_str = ';'.join(ele)
                    IFP_file.write(write_str + '\n')
                IFP_file.close()
        else:
            IFP_file = open(IFP_filename, 'a')
            # TODO: When the new models are available, the value for 
            # score would can be added
            """
            content = \
                f'{r_pdbid};{select_db};{id_proname[r_pdbid]};' + \
                f'{id_ligname[r_pdbid]};{r_ligandscore};n.a.;' + \
                f'{r_ligandscore};' + str(score) + ';' + \
                f'{r_dockscore}\n'
            """
            content = \
                f'{r_pdbid};{select_db};{id_proname[r_pdbid]};' + \
                f'{id_ligname[r_pdbid]};{r_ligandscore};n.a.;' + \
                f'{r_ligandscore};' + f'{r_dockscore}\n'
            IFP_file.write(content)
            IFP_file.close()
    #### PART4 COMPLETED ####

    os.system('rm input.pdb')       # Remove the file copied to fix: AssertionError , "%s does't exist" %filename

    status_f.write(f'{input_num}:{select_db}:Complete\n')
    status_f.close()

    print('DONE')
except Exception as e:
    status_f = open('Output/status.txt', 'a')
    status_f.write(f'{input_num}:{select_db}:Fail\n')
    status_f.close()
    print('NO RESULT')
