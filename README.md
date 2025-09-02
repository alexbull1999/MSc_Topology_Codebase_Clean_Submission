# Topology of Language for Semantic Entailment

This codebase implements topological data analysis on natural language inference (NLI) datasets. The work demonstrates that logically distinct entailment classes (entailment, neutral, contradiction) have materially different topological signatures, that can be used to improve classification and clustering performance on key NLI datasets, and that better reflect human judgements of ambiguous premise-hypothesis pairs than current models.

The research makes the following novel contributions:
- We prove that topological differences exist on a class-level between logically different semantic entailment relationships; establishing the existence of a `Topology of Language'. We demonstrated the strength of the topological differences between entailment, neutral, and contradiction classes achieving 100\% clustering accuracy across several distance metrics and embedding spaces.
- We show how topological differences can be extended from a class-level to an individual sample-level to achieve state-of-the-art classification accuracies on key NLI datasets. We exploit the hierarchical and asymmetric nature of entailment relationships by using Order and Asymmetry models to generate sufficient data points per premise and hypothesis pair for stable persistent homology analysis. Using K-Means Clustering on the $H_0$ and $H_1$ persistence diagrams from the persistent homology analysis allowed us to achieve unsupervised classification scores of 70.8\%, 59.7\%, and 60.6\% on SNLI test, MNLI matched, and MNLI mismatched datasets respectively. These scores beat previous benchmarks by a between 5-10\%.
- We demonstrate how our topological method captures fundamental information about entailment relationships that traditional NLI models miss, and which allows us to more accurately predict the distribution of human judgements on ambiguous entailment cases. The nature of gold-label classification tasks as a means of evaluating a model's NLI capabilities is flawed given the inherent ambiguity and subjectivity at the heart of language. Instead, predicting the distribution of human opinions - as the ChaosNLI challenge asks models to do - is a superior means of assessment. We demonstrate how a CNN trained on persistence images from our topological method materially improves the performance of all existing SOTA models' performance on ChaosNLI, whether measured by JSD or by KL divergence. 
- We prove that topological differences between entailment classes can be preserved in an extremely compressed latent space. We train a variety of autoencoders to take the input 1536D embeddings that achieved $\sim$100\% clustering accuracy, and compress them by over 20 times, to a 75D latent space. We show how several of our autoencoder variations are able to maintain to a high degree of accuracy the topological signatures of the input embeddings, with these best performing autoencoders achieving 98\% accuracy on an identical clustering task using the 75D latent embeddings. 
- We discover that the topology of entailment classes is preserved organically from information-theoretic optimisation objectives, rather than requiring explicit topological guidance. We observe how a purely reconstruction loss-driven autoencoder, with no purpose-built loss functions aimed at topological preservation, manages to preserve the topology of our original embeddings better than all other autoencoder variations. This finding implies a deeper connection between topology and semantics than expected. Namely, it demonstrates that there is a strong link between the topological signatures of entailment classes and the underlying semantic structure of said classes, since the topology was preserved purely from an autoencoder attempting to create the most efficient encodings of the input embeddings. 


### Environment Setup

##### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 64GB+ RAM for large-scale experiments

##### Installation

```
# Clone the repository
git clone git@github.com:alexbull1999/MSc_Topology_Codebase_Clean_Submission.git
cd MSc_Topology_Codebase

# Create conda environment
conda env create -f environment.yml
conda activate tda_entailment_new

# Set up Python paths
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Download datasets
python data/download_datasets.py 

# Extract arrow files into json files (repeat the below for snli train, val, and test, datasets and equivalent for mnli)
python data/extract_snli_full.py 
python data/extract_mnli_full.py 

```

### Experimental Workflows:

##### 1. Class-Level Persistence Image Clustering Experiments

```
# Generate SBERT embeddings
python entailment_surfaces/text_processing_sbert.py

# Train Order Embedding model
python src/order_embeddings_asymmetry.py

# Run the various clustering experiments (Choosing the dataset you want to run the clustering experiment on (e.g., MNLI, SNLI) in the code files themselves)

# For comparing Persistent Homology Dimension values across entailment classes for different distance metrics and embeddings spaces (Ch3: Experiment 1 in Thesis)
python entailment_surfaces/phdim_distance_metric_optimized.py 

# For running a single clustering test on the persistence images of 10 random samples of 1000 entailment, neutral, and contradiction points; using a range of different distance metrics and embedding spaces
python entailment_surfaces/phdim_clustering.py

# For running 10 clustering tests, each on the persistence images of 10 random samples of 1000 entailment, neutral, and contradiction points; using a range of different distance metrics and embedding spaces. This provides a more statistically significant clustering answer, used in Ch3: Experiment 2 in Thesis
python entailment_surfaces/phdim_clustering_validation_v2.py
```

##### 2. Sample-Level Persistence Image Clustering and Classification Experiments

###### Unsupervised K-Means Clustering Experiments

```
# Extract hidden-layer token representations from SBERT model
python phd_method/src_phd/sbert_token_extractor.py  # Note sbert_token_extractor_chunked.py can help with OOM issues

# Train Order and Asymmetry models
python phd_method/src_phd/independent_order_model.py 
python phd_method/src_phd/independent_asymmetry_model.py 

# Run unsupervised clustering tests on validation set; adjusting boundaries of stratified, and energy-weighted features to optimize for the validation set (optimal boundaries already preset). Then run unsupervised clustering tests on test sets to get final results
# Note, use the SNLI Order and Asymmetry models for the SNLI persistence image extraction, and vice-versa
python phd_method/src_phd/point_cloud_clustering_SNLI.py
python phd_method/src_phd/point_cloud_clustering_MNLI.py

# To get the comparative SBERT baseline clustering results (non-topological), for which we use the InferSent feature extraction methodology
python phd_method/src_phd/point_cloud_clustering_saved_pre_classification_vSBERT_Baseline.py

```

###### Supervised Persistence Image CNN Classification Experiments

```
# Precompute Persistence Images for Efficiency (Repeat for SNLI Train, Val, Test, and equivalent with MNLI)
# Note, use the SNLI Order and Asymmetry models for the SNLI persistence image extraction, and vice-versa
python phd_method/src_phd/precompute_persistence_images_chunked.py 

# Run classification for SNLI and MNLI
python phd_method/src_phd/final_CNN_persim_SNLI_classifier_ALL_TRAIN_CHUNKS.py
python phd_method/src_phd/final_CNN_persim_MNLI_classifier_ALL_TRAIN_CHUNKS.py

```

##### 3. ChaosNLI JSD and KL performance

```
# Requires cloning of the ChaosNLI GitHub repo for the baseline original model predictions: https://github.com/easonnie/ChaosNLI 

# Extract SBERT tokens images on ChaosNLI datasets
python phd_method/src_phd/chaosNLI_sbert_token_extractor.py

#Compute persistence images
# Note, use the SNLI Order and Asymmetry models for the ChaosNLI-SNLI persistence image extraction, and vice-versa for ChaosNLI-MNLI
python phd_method/src_phd/chaosNLI_persistence_image_precompute.py

# Perform weighted average predictions between persistence image CNN and original ChaosNLI model predictions:
# Needs to be performed from home directory to have access to ChaosNLI cloned directory 
cd ~ 
python MSc_Topology_Codebase_Clean_Submission/phd_method/src_phd/ALL_TRAIN_CHUNKS_chaosNLI_hybrid_SOTA_PersimCNN_image_classification_SNLI_MNLI_separate_eval.py

```


##### 4. Topology of 75D Autoencoder Latent Space vs Input 1536D Embedding Topology

```
# Create prototype persistence diagrams of input embedding topology (Note, you can select distance metric (e.g. cosine), and prototype method (e.g., bottleneck))
python entailment_surfaces/supervised_contrastive_autoencoder/src/persistence_diagram_prototype_creator.py

# Train autoencoders
# For pure reconstructive autoencoder, contrastive autoencoder, or combinations of these two only 
# Adjust the reconstruction and contrastive weights in the config in full_pipeline_global.py per preference
python entailment_surfaces/supervised_contrastive_autoencoder/src/full_pipeline_global.py

# For autoencoders with topological loss functions:
# Edit  entailment_surfaces/supervised_contrastive_autoencoder/src/losses_global_topological.py, to use the correct topological loss function desired (e.g. moor vs. gromov-wasserstein)
# Then run the below file, adjusting relative loss function weights in the config per preference:
python entailment_surfaces/supervised_contrastive_autoencoder/src/train_topological_autoencoder.py 


# To analyse and compare latent topology with input embedding topology; select the appropriate input topology prototype and latent autoencoder you want to compare in the below file and run it
python entailment_surfaces/supervised_contrastive_autoencoder/src/persistence_diagram_comparison_inputvslatent.py

```
