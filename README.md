# N Body Problem

Python solution and plotting of the gravitational N-body problem
Scientific Computation Project - University Of Nottingham

## Structure
```
NBody
├── Figure8.ipynb (Jupyter: animations of for Figure Of Eight Problem)
├── README.md
├── bodies (.json files with initial values)
│   ├── *.json
├── caching.py (Helper functions to cache results)
├── data (Cached results)
│   ├── *_t (Files of this format store time values)
│   ├── *_y (Files of this format store position and speed values)
├── index.json (List of hashes that have been cached)
├── nbody.py (Main program to be run)
├── physics.py (Physical calculations)
├── radial_velocity_HD2039.ipynb (Jupyter: Radial velocity of a star)
├── radialdata
│   └── radialdata.sh (Downloads real radial velocity data using wget)
├── report (Latex report for this project)
│   ├── main.tex
│   ├── physics_article_B.cls (Preamble for main.tex)
│   └── references.bib (Generated references from Zotero)
├── requirements.txt (Generated from pip freeze)
└── tools.py (Useful functions used in development)
```

## Requirements

Using a virtual python environment with Py > 3.7 run
```
pip install -r requirement.txt
```
to install the dependencies

The crucial packages and the version used here are:
- numpy==1.17.2
- scipy==1.3.1
- jupyter

## Usage

The Jupyter Notebooks give some examples of how `nbody.py` is used.

`nbody.py` can be run directly using python.
It will promp for a file name, provide the json file you want to load excluding the file extension.
The following occurs:

1. The initial conditions will be loaded
2. The trajectories will be calculated
3. The trajectories will be cached so that the same initial conditions need not be re-calculated
4. The first figure window plots the trajectories.
  - Options to this are exposed in the draw_bodies function call in nbody.py
5. The second figure window plots the total angular momentum and total energy of then system at each time value

## Report

The report is generated using XeLaTeX and the (open source) font (Lora)[https://fonts.google.com/specimen/Lora].
References were generated using Zotero
