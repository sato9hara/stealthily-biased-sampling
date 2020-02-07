# stealthily-biased-sampling

Python code for influential instance estimation proposed in the following paper.
* K. Fukuchi, S. Hara, T. Maehara, [Faking Fairness via Stealthily Biased Sampling](https://arxiv.org/abs/1901.08291). to appear in AAAI'20 Special Track on Artificial Intelligence for Social Impact (AISI).

## How to run 
Before running Jupyter Notebook files, please follow the steps below.

### Requirements ###
- g++ (-std=c++11)
- [lemon](https://lemon.cs.elte.hu/trac/lemon/)

### 1. Install LEMON ###

```
sudo apt install g++ make cmake 
wget http://lemon.cs.elte.hu/pub/sources/lemon-1.3.1.tar.gz
tar xvzf lemon-1.3.1.tar.gz
cd lemon-1.3.1
mkdir build
cd build
cmake ..
make
sudo make install
```

### 2. Make files ###

```
cd stealth-sampling
make
cd ../wasserstein
make
```
