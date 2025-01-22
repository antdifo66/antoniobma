# antoniobma

This is a project I am completing for my W2025 co-op research term. The goal of this project is for me to get an understanding of Bayesian Model Averaging, using simple polynomial models and data.

Most of this project, specifically the part of the code that runs the BMA analysis on the data, was written by Simone Paradiso. This repo can be found here: https://github.com/simonpara/Fast-MPC

This project contains a couple .ipynb project files, however these have dependencies on other project files. Some of them, like polynomialbma.py and FastMPClinpara.py, were either created by me or based almost entirely on Simone Paradiso's FastMPC code, you can download those from this repo. Which one of these specifically is required for a particular .ipynb will be listed at the top of the file. Some are Python libraries, like Cobaya, GetDist, etc, which must be installed separately.

It should also be noted that these files contain addresses to input/output files from/to, which currently only point to addresses within my own device. These will cause issues if this code is run on another device, of course, so be sure to change these. Usually running the code will create an error message that points to where in the code this address is typed, so they won't be too hard to find. Pretty much all of the addresses used in the code I am providing in this repo point to a single folder on my device called "BMA_stuff", so replacing the content of the address before and including this folder wth the address of the folder you want to put these files and their output in should work.
