import sys
import csv
from operator import itemgetter


DIR = sys.argv[1]
MIN_LIG_SIM = float(sys.argv[2])
MIN_BIN_SIM = float(sys.argv[3])
MIN_LTM_SIM = float(sys.argv[4])

matches = []

fields = ['PDB', 'Class', 'TargetName', 'LigandName', 'LigandSimilarityScore', 'BindingSimilarityScore', 'LigTMapScore', 'DockingScore']
reader = csv.DictReader(open(f'{DIR}/IFP_result.csv'), fieldnames=fields, delimiter=';')

for i in reader:
    try:
        if  float(i['LigTMapScore']) >= MIN_LTM_SIM or \
            float(i['BindingSimilarityScore']) >= MIN_BIN_SIM or \
            float(i['LigandSimilarityScore']) >= MIN_LIG_SIM:
            matches.append(i)
    except ValueError:
        ...

bests = sorted(
    matches, key=itemgetter('LigTMapScore', 'BindingSimilarityScore', 'LigandSimilarityScore'), reverse=True)

writer = csv.DictWriter(open(f'{DIR}/bests.csv', 'w'), fieldnames=fields, delimiter=';')
writer.writeheader()
for i in matches:
    writer.writerow(i)
