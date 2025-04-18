# MesoNet V1.5: Optimizing Spiking Neural Networks with Dynamic Saddle Distributions

**NOTE (MesoNet) [Updated 4/3/2025]:** Currently updating with MesoNet V2 and organizing this repo for public use!

## <h2 id="about">About (CSEF Abstract)</a> </h2>
Artificial neural network (ANN) clustering algorithms require hundreds of watts, considerably limiting their abilities across numerous energy-critical applications. A developing technology, spiking neural networks (SNNs) offer a promising alternative, requiring mere milliwatts by mimicking biological neurons on neuromorphic chips. However, the prevalent spike-timing-dependent-plasticity (STDP) learning rule struggles to scale for practical tasks, reducing accuracy by up to 40% compared to ANNs. This project addresses these challenges with a two-step approach: mathematically investigating SNN learning behavior with dynamical systems theory, before developing an algorithm for improved generalization.

To disassociate the STDP’s recursive nature, I approximated the LIF neuron kernel via restricted quadratic form. Expanding to the multilayer case, I discovered that weight updates can be described as a stability-switch bifurcation regulated by learning rate. Furthermore, I proved an optimal learning regime exists when network parameters operate at the edge of chaos. Leveraging these discoveries, the MesoNet architecture introduces two novel algorithms– variable plasticity and triangulated attribution– maintaining optimal learning conditions. These mechanisms dynamically form and annihilate saddle points to enhance representations. Notably, a Turing map of inhibitory-excitatory interactions displayed emergence of cortical-like structures, motivating a split-and-merge architecture.

On empirical tests, MesoNet achieves phenomenal performance in accuracy and information retention. This includes achieving a 56.6% accuracy on CIFAR-10, 18.57% points above the previous SOTA SNN. MesoNet lies within 5.05% points of state-of-the-art ANNs while consuming half (47.23%) the energy. This architecture demonstrates significant potential for transforming computation in applications including deep-space exploration, medical implants, and remote sensing.

## <h2 id="authorsnote">Author's Note</a> </h2>
MesoNet was created through the individual research of Krishna Bhatt. While researching neuroimaging last year, I came across bio-plausible neuron simulations. I was fascinated by its possible computational applications and individually began work on this year’s project in July 2024, after a literature review.

## <h2 id="organization">Repository Organization</a> </h2>
#### Requirements
- Python 3.11.8
  - Brian2
  - Numpy
  - Scipy
  - Pickle
  - Heapq

#### Directory Structure
- `root\assets`: Random images, visualizations, etc. from the project
- `root\benchmarks`: Benchmark datasets used for testing (model is based upon MNIST)
- `root\experimentation`: Workspace with random assortment of ideas, testing, and experimentation during development
- `root\model`: MesoNet V1.5 (currently polishing up V2)
  - `root\model\functions.py`: Input encoding, MesoNet learning algorithms, clustering and assignments
  - `root\model\mesonet.py`: Primary model structure (split and merge)
  - `root\model\train.py`: Simulation controller
  - `root\model\utils.py`: Utils

