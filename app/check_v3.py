import numpy as np
import langchain
from pydantic import VERSION as pydantic_version

print(f"Python version 3.13 detected")
print(f"NumPy version: {np.__version__}")
print(f"Langchain version: {langchain.__version__}")
print(f"Pydantic version: {pydantic_version}")