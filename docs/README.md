# Documentation

This project supports automatic documentation generation via the [sphinx](https://www.sphinx-doc.org/en/master/) documentation builder.

To build the documentation, execute the following commands in the root directory
```
pip install .
pip install MCL_PPL_Benchmark[docs]

cd docs
make [html, latex] # choose the desired documentation. 
```