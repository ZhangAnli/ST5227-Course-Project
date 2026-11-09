import os,sys
os.environ.setdefault('MPLCONFIGDIR', '/tmp/st5227-density-mpl')
os.environ.pop('MPLBACKEND', None)
from pathlib import Path
import nbformat
from nbclient import NotebookClient
root=Path(__file__).resolve().parent
path=root/'Project-YiHern-density.ipynb'
n=nbformat.read(path,4)
# Leave the optional unsupervised section unexecuted.
stop=len(n.cells)
optional=n.cells[stop:]; n.cells=n.cells[:stop]
smoke='--smoke' in sys.argv
if smoke:
 for c in n.cells:
  if 'results, result_directory = run_experiments' in c.source:
   c.source=c.source.replace('NUMBER_OF_EXPERIMENTS = 20','NUMBER_OF_EXPERIMENTS = 2').replace('N_JOBS = 4','N_JOBS = 2')
   c.source=c.source.replace('start = time.perf_counter()',"X = X.sample(n=240, random_state=42)\nY_regression = Y_regression.loc[X.index]\nstart = time.perf_counter()")
   c.source=c.source.replace('n_jobs=N_JOBS)', 'n_jobs=N_JOBS, smoke=True)')
output=root/'density-results'/'smoke-preview.ipynb' if smoke else path
output.parent.mkdir(exist_ok=True)
client=NotebookClient(n,timeout=None,kernel_name='st5227-project',resources={'metadata':{'path':str(root)}})
def on_complete(cell,cell_index,**kwargs):
 print('Completed cell',cell_index,flush=True)
 nbformat.write(n,output)
client.on_cell_executed=on_complete
try:
 client.execute()
finally:
 n.cells += optional
 nbformat.write(n,output)
print('PASS', output,flush=True)
