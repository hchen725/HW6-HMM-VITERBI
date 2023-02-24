.. hw6-hmm documentation master file, created by
   sphinx-quickstart on Sat Feb 11 16:27:34 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Lab 6: Inferring CRE Selection Strategies from Chromatin Regulatory State Observations using a Hidden Markov Model and the Viterbi Algorithm
============================================================================================================================================

The aim of hw6 is to implement the Viterbi algorithm, a dynamic program that is a common decoder for Hidden Markov Models (HMMs). The lab is structured by training objective, project deliverables, and experimental deliverables:

**Training Objective**: Learn how to design reusable Python packages with automated code documentation and develop testable (user case) hypotheses using the Viterbi algorithm to decode the best path of hidden states for a sequence of observations.

**Project Deliverable**: Produce a simple report for functional characterization inferred from a binary regulatory observation state pattern across cardiac developmental timepoints.

**Experimental Deliverable**: Construct a positive control library for massively parallel reporter assays (MPRAs) and CRISPRi/a experiments in primitive and progenitor cardiomyocytes (i.e., cardiogenomics).

Key Words
==========
Chromatin; histones; nucleosomes; genomic element; accessible chromatin; chromatin states; genomic annotation; candidate cis-regulatory element (cCRE); Hidden Markov Model (HMM); ENCODE; ChromHMM; cardio-genomics; congenital heart disease(CHD); TBX5


Functional Characterization Report
===================================

Please evaluate the project deliverable and briefly answer the following speculative question, with an eye to the project's limitations as related to the theory, model design, experimental data (i.e., biology and technology). We recommend answers between 2-6 sentences. It is OK if you are not familiar already with this biological user case; you can receive full points for your best-effort answer.

1. Speculate how the progenitor cardiomyocyte Hidden Markov Model and primitive cardiomyocyte regulatory observations and inferred hidden states might change if the model design's sliding window (default set to 60 kilobases) were to increase or decrease?
Observations and hidden states might change with the sliding window size because it would impact the accessible chromatin regions that are captured in the window by the two CRE selection method. For example, encode_atac might capture a greater number of accessible chromatin regions that are more dispersed across a larger window while atac captures regions that are more localized to a region, it might influence the outcome observation state. 

2. How would you recommend integrating additional genomics data (i.e., histone and transcription factor ChIP-seq data) to update or revise the progenitor cardiomyocyte Hidden Markov Model? In your updated/revised model, how would you define the observation and hidden states, and the prior, transition, and emission probabilities? Using the updated/revised design, what new testable hypotheses would you be able to evaluate and/or disprove?
One would probably start with reading up on these modalities and gaining a better understanding of how ChIP-seq data influences the observed states - are regulator and regulator potential the only two outcomes, are there additional observations? Assuming the same two possible observation states, a transition probability matrix would require including probabilities of each data source transitioning to the other. The emission probabilities would continue to have two observed states, but will need to contain probabilties for all four sources of genomics data. 

3. Following functional characterization (i.e., MPRA or CRISPRi/a) of progenitor and primitive cardiomyocytes, consider all possible scenarios for recommending how to update or revise our genomic annotation for *cis*-candidate regulatory elements (cCREs) and candidate regulatory elements (CREs)?
Sounds like a lot of possible scenarios to consider given the vast array of perturbations that can be conducted with MPRA, CRPSIRi/a (not even including other CRISPR technologies in the work, CAS9 vs CAS12 considerations), etc. It would be important to understand how the annotation of cCREs and CREs will change in different functional contexts as the annotations might switch in these contexts. I would probably start by stating the functional context that I'm starting with and selecting for the assays that are relevant to that context rather than trying to model all of them at once. 

Models Package 
======================
.. toctree::
   :maxdepth: 2
   
   modules
