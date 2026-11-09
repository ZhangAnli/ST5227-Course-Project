"""Execute the both-grades square-root notebook, preserving completed outputs."""
import os, sys
os.environ.setdefault('MPLCONFIGDIR', '/tmp/st5227-sqrt-both-mpl')
os.environ.pop('MPLBACKEND', None)
from pathlib import Path
import nbformat
from nbclient import NotebookClient
root = Path(__file__).resolve().parent
path = root/'Project-YiHern-sqrt-both.ipynb'
notebook = nbformat.read(path, 4)
smoke = '--smoke' in sys.argv
if smoke:
    for cell in notebook.cells:
        if cell.cell_type == 'code' and 'NUMBER_OF_EXPERIMENTS = 20' in cell.source:
            cell.source = (cell.source.replace('NUMBER_OF_EXPERIMENTS = 20', 'NUMBER_OF_EXPERIMENTS = 2')
                .replace('N_JOBS = 4', 'N_JOBS = 2').replace('SMOKE = False', 'SMOKE = True'))
output = root/'sqrt-both-results'/'smoke-preview.ipynb' if smoke else path
output.parent.mkdir(exist_ok=True)
client = NotebookClient(notebook, timeout=None, kernel_name='st5227-project', resources={'metadata': {'path': str(root)}})
def on_complete(cell, cell_index, **kwargs):
    print('Completed cell', cell_index, flush=True)
    nbformat.write(notebook, output)
client.on_cell_executed = on_complete
try:
    client.execute()
finally:
    nbformat.write(notebook, output)
print('PASS', output, flush=True)
