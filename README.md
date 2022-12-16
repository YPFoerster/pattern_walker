<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>

# Pattern Walker

The main module implements the patternwalker model (simplified and with full set of parameters). Submodules incorporate several utilities, wrappers for plotting
trees and derived patternwalker classes that compute the complexity $C$ associated with the given set of parameters. Patternwalker classes are derived from networkx.DiGraph.

##Installation

Install using Python pip, running
```
pip install git+https://github.com/YPFoerster/pattern_walker
```
for the latest version.

## Usage

Some basic commands involving the main classes and module functions. Head over to [examples](examples/) to see them in application.
The Jupyer notebook referred to in my thesis can be found in the folder [inference](inference/).

### Creating a patternWalker instance

First create rooted tree with edges directed away from the root (this is imporant as it defines a hierarchy on the nodes). For instance, use the utils function balanced_ditree to make a $c$-ary tree of height $h$ with directed edges
```python
tree,root = utils.balanced_ditree(c,h)
```
This hierarchy serves as a "background" for the patternWalker:
```python
Walker=fullProbPatternWalker(tree, root, L, a_root, a_low, a_high, Delta, Gamma, Gamma_root)
```
\texttt{fullProbPatternWalker} is the full model considered in the [corresponding article](http://dx.doi.org/10.1088/2632-072X/ac8e48); the more primitive class \texttt{patternWalker} takes fewer parameters.
The parameters are L and Delta are nonegative and integer, and the remaining parameters can be between $0$ and $1$. Moreover, a_low and a_high should obey

$$
\begin{align}
a_{\rm low}<&a_{\rm high},\\
(1-a_\rm{root}) \cdot \rm{Gamma}_\rm{root}<&a_\rm{low},a_\rm{high}<(1-a) \cdot \rm{Gamma}_\rm{root} + a_\rm{root} \\
\end{align}
$$

Currently, patterns and edge weights are not set automatically! ~Wash thoroughly~ Run
```python
Walker.set_weights()
```
before use.

### Basic functionality

#### walker

The fundamental class for all patternWalkers is the walker class. With initial position at the node passed as root, we can have the walker do a step to a random neihbour (probabilities proportional to edge weights) using
```python
Walker.step()
```
The current position and number of steps done are stored in
```python
Walker.x
```
and
```python
Walker.t
```
respectively. The walker can be reset to the initial position, and x and t to 0, using
```python
Walker.reset_walker()
```

#### patterns

Along with
```python
Walker.set_weights()
```
we can produce a new realisation of patterns calling
```python
Walker.reset_patterns()
```
which also re-calculates the edge weights according to the new patterns.

#### Calculating MFPTs

The utils-module has a function to calculate MFPTs for a list of given node pairs, based on the "grounded Laplacian" method

$$
\begin{align}
m_{ij} = (\mathbf{I} - \widehat{\mathbf{W}}_{j})^{-1}\mathbf{e}
\end{align}
$$

where $\mathbf{I}$ is the unit matrix, $\widehat{\mathbf{W}}_{j}$ is the transition matrix of the walker with row and column of the target-node $j$ removed, and $\mathbf{e}$ is the all-one vector.
To use this function, call
```python
utils.mfpt(Walker,[(source, target)])
```
Most commonly, you will want to call
```python
utils.mfpt(Walker,[(Walker.root, Walker.target_node)])
```
The list of source-target pairs can be longer, but the function currently does not check if the same target appears several times (which would be preferrable because it would reduce the number required matrix inversions).

#### Approximate Complexity

The approximate complexity $C$ in all its glory is hidden in the mean_field-module. With the same parameters as above, call
```
Walker=overlap_MF_patternWalker(tree, root, L, a_root, a_low, a_high, Delta, Gamma, Gamma_root)
```
The class is derived from fullProbPatternWalker, so it has all the basic functionality. Additionally,
```python
Walker.MF_mfpt()
```
returns the complexity $C$.

## Requirements

gensim==4.1.2
matplotlib==3.5.1
naming==0.7.0
networkx==2.6.3
nltk==3.7
numpy==1.21.4
pandas==1.3.4
scikit_learn==1.1.3
scipy==1.7.3
spacy==3.4.3
