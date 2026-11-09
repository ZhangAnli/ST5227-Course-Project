"""Execute the polynomial comparison, saving outputs after every completed cell."""
import os, sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/st5227-polynomial-mpl')
os.environ.pop('MPLBACKEND',None)
from pathlib import Path
import nbformat
from nbclient import NotebookClient
root=Path(__file__).resolve().parent
path=root/'Project-YiHern-polynomial.ipynb'
n=nbformat.read(path,4)
smoke='--smoke' in sys.argv
if smoke:
    for cell in n.cells:
        if cell.cell_type != 'code':
            continue
        cell.source=cell.source.replace('NUMBER_OF_EXPERIMENTS = 20','NUMBER_OF_EXPERIMENTS = 2').replace('N_JOBS = 4','N_JOBS = 2').replace('SMOKE = False','SMOKE = True')
        if 'reference_results, reference_directory' in cell.source:
            cell.source='''# Include rare low grades to exercise subgroup plots in the small pilot.
pilot_index = df[df.grade_class.ne('F')].sample(n=200,random_state=42).index.union(
    df[df.grade_class.eq('F')].sample(n=40,random_state=42).index)
X = X.loc[pilot_index]
Y_regression = Y_regression.loc[pilot_index]
''' + cell.source
output=root/'polynomial-results'/'smoke-preview.ipynb' if smoke else path
output.parent.mkdir(exist_ok=True)
client=NotebookClient(n,timeout=None,kernel_name='st5227-project',resources={'metadata':{'path':str(root)}})
def on_complete(cell,cell_index,**kwargs):
    print('Completed cell',cell_index,flush=True)
    nbformat.write(n,output)
client.on_cell_executed=on_complete
try:
    client.execute()
finally:
    nbformat.write(n,output)
print('PASS',output,flush=True)
