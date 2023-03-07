How to build the conda package
===============================

```bash
conda install anaconda-client conda-build
conda build conda-recipe -c defaults -c conda-forge -c isarandi

anaconda login
anaconda upload .../poseviz-...tar.bz2
```

## References

- https://docs.conda.io/projects/conda-build/en/latest/user-guide/tutorials/build-pkgs.html
- https://blog.gishub.org/how-to-publish-a-python-package-on-conda-forge
- https://www.underworldcode.org/articles/build-conda-packages/
- https://levelup.gitconnected.com/publishing-your-python-package-on-conda-and-conda-forge-309a405740cf
- https://python-packaging-tutorial.readthedocs.io/en/latest/conda.html