import platform
import sys

def get_version(package_name):
    try:
        module = __import__(package_name)
        return getattr(module, '__version__', 'unknown')
    except ImportError:
        return 'not installed'


print('--- System Information ---')
print(f'Platform:   {platform.platform()}')
print(f'Python:     {sys.version}')
print(f'omop-graph:   {get_version("omop_graph")}')
print(f'omop-emb:   {get_version("omop_emb")}')
print(f'pgvector:   {get_version("pgvector")}')
print(f'FAISS CPU:    {get_version("faiss-cpu")}')
print(f'FAISS GPU:    {get_version("faiss-gpu")}')
print(f'NumPy:      {get_version("numpy")}')
print(f'SQLAlchemy: {get_version("sqlalchemy")}')
print('--------------------------')