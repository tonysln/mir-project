
# MIR Project

**Course:** [LTAT.02.015] Music Information Retrieval 2023/24, UT

**Topic idea**: Vocal track extraction / instrument separation toolset

**Goal:** Create a simple to use(!) program for extracting specific instruments from a song and cleaning them up to sound passable. Must support common pop/rock instruments, such as vocals, guitars, drums (at least partly), and bass. Potentially allow mixing and matching various songs and instruments.

**Setup using conda environment:**

1) Download the created git project
2) Move into the MIR-PROJECT folder on command line
3) conda create -n "mir" python==3.10
4) conda activate mir
5) pip install -r requirements.txt

Test the program using mixing.ipynb notebook following the instructions given in the notebook. 

**Testing the results:**
Our main results can be tested in the mixing.ipynb notebook following the instructions given in the notebook. Main idea is that you can provide YouTube links for two songs and mix them together. It is possible to extract the vocals, backtrack and percussion. It is also possible to layer different songs together (backtrack from first song, vocals from another). For that the key of the base backtrack is detected and the other vocal part is shifted to fit the base key. It is also possible to unify the tempo. Different options are given with sliders - for example for selecting only a part of a song and controlling the volumes of different layers.

**Team:** Anton Slavin, Ines Anett Nigol



---

Used materials:

- https://kinwaicheuk.github.io/nnAudio/index.html
- https://github.com/tsurumeso/vocal-remover
- https://towardsdatascience.com/finding-choruses-in-songs-with-python-a925165f94a8