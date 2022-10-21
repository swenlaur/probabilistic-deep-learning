## Legend for book sources
* **PDL:** Probabilstic Deep Learning models
* **SR:** Statistical Rethinking: A Bayesian Course with Examples in R and Stan (Second Edition)  
* **TF** Machine Learning with Tensor Flow

## Classical Deep Learning

### Deep Learning Frameworks (Sven Laur)

Concepts
* Hardware model behind Deep learning
* Tensorflow
* Pytorch

Additional materials
* ??

### Feed-Forward neural network architectures (PDL 25-61, TF 276-294)

Concepts
* Fully connected neural networks
* RELU and sigmoid neurons
* Max pooling
* Convolutional neural networks

Additional materials
* ??


### Recurrent neural network architectures (TF 335-364)

Concepts
* LSTM

Additional materials:
* [C. Olah. Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs)
* [S. Hochreiter and J. Schmidhuber. Long Short-term Memory](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory)


### Optimisation methods (PDL 63-90)

Concepts
* Learning rate
* Gradient decent
* Stochastic gradient decent
* Automatic differentation
* Static and dynamic backpropagation
* Hyperparameter optimisation

Additional materials
* ??

### Maximum likelihood and loss functions (PDL 91-127)

Concepts
* Maximum likelihood
* Maximum aposteriori probability
* Loss function derivation
* Crossentropy and negative log-likelihood
* Kullback-Leibler divergence
* Mean square error and negative log-likelihood

Additional materials about the interpretation of Kullback-Leibler divergence
* [Sanjeev R. Kulkarni. Information, Entropy, and Coding](https://www.princeton.edu/~cuff/ele201/kulkarni_text/information.pdf)
* [Zachary Robertson. KL Divergence as Code Patching Efficiency](https://www.lesswrong.com/posts/42yPvyppEoBdQZsp3/kl-divergence-as-code-patching-efficiency)
* [Will Kurt. Kullback-Leibler Divergence Explained](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)


## Attaching probability to outcomes

**Main method:**
Use data to fix a single model and use the model to assign probabilities to observations.
Does not work well if there are many near optimal models with widely different predictions.


### Probabilistic models for continuous data (PDL129-145)

Concepts
* Negative log-likelihood as loss
* Prediction intervals from predicted distribution
* Models with varying error terms aka Heteroskedasticity
* Predicting mean and variance

Additional materials
* ??

### Probabilistic models for count data (PDL 145 - 156, SR 323-380)

Concepts
* Binomial distribution
* Poisson distribution
* Zero-Inflated Poisson distribution
* Negative binomial distribution
* Logistic regression
* Poisson regression
* Diagnostic methods

Additional materials
* [E. Cotner. od Spiked the Integers with pyro](https://github.com/ecotner/statistical-rethinking/blob/master/notebooks/11_god_spiked_ints.ipynb)


### Mixture models (PDL 157-166, SR 359-366, 369-397)

Concepts
* Multinomial distribution
* Discretised logistic mixture distribution
* Regression with discretised logistic mixture distribution
* WaveNet and PixelCNN

Additional materials
* [DeepMind Blog. WaveNet: A generative model for raw audio](https://www.deepmind.com/blog/wavenet-a-generative-model-for-raw-audio)
* [A. Oord et al. WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)
* [A. Moussa. A Guide to Wavenet ](https://github.com/AhmadMoussa/A-Guide-to-Wavenet)
* [A. Oord et al. Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759)
* [T. Salimans et al. PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications](https://arxiv.org/abs/1701.05517)


### Normalising flows (PDL 166 - 193)
* Transformation functions (bijectors)
* Probability density function and Jacobian
* Maximum likelihood to estimate parameters of Transformations
* Neural networks as Transformation functions
* Glow model for faces

Additional materials
* [Pyro. Normalizing Flows - Introduction](https://pyro.ai/examples/normalizing_flows_i.html)
* [Open AI blog. Glow](https://openai.com/blog/glow/)
* [D. P. Kingma, P. Dhariwal. Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/abs/1807.03039)


## Working with posteriors

**Main method**
Use data to fix the weight of individual models and use these to average over predictions.
Allow to measure the uncertainty due to variability of training data.   

### Basics of Bayesian inference (PDL 197-228)

Concepts
* Model averaging
* Bayes formula and corresponding inference rules
* Coin-tossing example
* Bayesian linear regression model

Additional materials
* [Pyro: Bayesian Regression: Part I](https://pyro.ai/examples/bayesian_regression.html)
* [Pyro: Bayesian Regression: Part II](https://pyro.ai/examples/bayesian_regression_ii.html)

### Variational inference (PDL 229-245)

Concepts
* Kullback-Leibler difference
* Parametric posterior approximation
* Variational inference for a single neuron
* Bayes Backprop algorithm in practice
* Stochastic Variational Inference

Additional materials
* [Blundell et al. Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424)
* [Jospin et al. Hands-on Bayesian Neural Networks - A Tutorial for Deep Learning Users](https://arxiv.org/pdf/2007.06823.pdf)
* [An Introduction to Stochastic Variational Inference in Pyro](https://pyro.ai/examples/svi_part_i.html)

### Monte-Carlo dropout (PDL 245-263)

Concepts
* Dropout layer as a regulariser
* Monte-Carlo dropout architecture
* Uncertainty measures for Bayesian classification

Additional materials
* [Yarin Gal, Zoubin Ghahramani. Dropout as a Bayesian Approximation](https://arxiv.org/abs/1506.02157)
* [Yarin. Dropout as a Bayesian Approximation. Source code](https://github.com/yaringal/DropoutUncertaintyExps)

### Markov-Chain-Monte-Carlo methods (SR 263-296)

Concepts
* Metropolis algorithm
* Gibbs sampling
* Hamiltonian Monte Carlo
* Adaptation, burn-in and convergence diagnostics

Additional materials
* [J. C. Orduz. A Simple Hamiltonian Monte Carlo Example with TensorFlow Probability](https://juanitorduz.github.io/tfp_hcm/)
* [H.-Y. Hu. Hamiltonian Monte Carlo in TensorFlow Probability](http://www.hongyehu.com/blog2-hamiltonian-monte-carlo-in-tensorflow-tutorial.html)
* [K. Sachdeva. Markov Chain Monte Carlo with Tensorflow Probability](https://github.com/ksachdeva/rethinking-tensorflow-probability/blob/master/notebooks/09_markov_chain_monte_carlo.ipynb)
* [E. Cotner. Markov Chain Monte Carlo with pyro](https://github.com/ecotner/statistical-rethinking/blob/master/notebooks/09_mcmc.ipynb)
