---
layout: page
title: Algorithms for Data Science 
description: Listing of course modules and topics.
mathjax: true
---

## <a name="Main-TextBooks"></a>Main TextBooks:

* [Data Mining and Analysis: Fundamental Concepts and Algorithms](https://dataminingbook.info/) by Mohammed J. Zaki and Wagner Meira Jr., 2021 (Data Mining & Analysis) [PDF](https://fumdrive.um.ac.ir/index.php/f/4160875)
* [Pattern Recognition](https://darmanto.akakom.ac.id/pengenalanpola/Pattern%20Recognition%204th%20Ed.%20(2009).pdf), by Sergios Theodoridis  and Konstantinos Koutroumpas, 2009
* [The Elements of Statistical Learning (ESL)](https://fumdrive.um.ac.ir/index.php/s/FH8nB4SwGkJrMeQ)
* [Foundations of Data Science (FDS)](https://www.cs.cornell.edu/jeh/book.pdf), by Avrim Blum, John Hopcroft, and Ravindran Kannan, 2018


## 1402/07/08
### <a name="L1"></a>Introduction to Data Science  
- Slide: [Introduction to Data Science](https://www.datasciencecourse.org/slides/15388_S22_Lecture_1_intro.pdf) by Zico Kolter
- Slide: [Introduction to Data Science](https://github.com/justmarkham/DAT8/blob/master/slides/01_intro_to_data_science.pdf) by Kevin Markham  
- Slide: [Clustering](https://mattdickenson.com/assets/clustering2.pdf) by Matt Dickenson 

**HW1**{: .label .label-red }[Generate random points with uniform distribution in the unit sphere](https://vu.um.ac.ir/mod/assign/view.php?id=441612), due date: 1402/07/21 (Extended)
    
### <a name="L2"></a>Python Programming
        
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)

**HW2**{: .label .label-red }[Satisfiability Table](https://vu.um.ac.ir/mod/assign/view.php?id=441616), due date: 1402/07/21 (Extended)


## 1402/07/10
Discrete Optimization
: Draft version of My Book: [Meta Heuristic Algorithms](https://www.dropbox.com/s/8bnxpzvfgiwma0k/combopt-PSO-20160514.pdf?dl=0)

Some Examples:
- N-Queen Problem
- Knight-Tour, [My old Delphi program](https://www.dropbox.com/s/okfa4lmzf1qqyus/NQueen-KnightTour.zip?dl=1), previous century!
- Traveling Salesman Problem
- Packing & Cutting Problems, [My old Delphi program](https://www.dropbox.com/s/3ztaqqlpki9e2ep/Thesis.zip?dl=1), previous century! My MSc. Project.

**Further Reading**{: .label .label-yellow }
- [Some published papers about the above programs](https://fumcs.github.io/projects/comb-opt/)

## 1402/07/15
Random Search
  : Chapter 1 & 2 of [My Book](https://www.dropbox.com/s/8bnxpzvfgiwma0k/combopt-PSO-20160514.pdf?dl=0) + [My NP-Complete Paper](https://www.dropbox.com/s/gdwuin9xycbvxwa/1379-npcomplete.pdf?dl=0)

**HW3**{: .label .label-red }[Python Code of Program 1.2, Page 11 of My book - RS](https://vu.um.ac.ir/mod/assign/view.php?id=449054&forceview=1), due: 1402/07/25

## 1402/07/17
**HW1-Sol**{: .label .label-green }, Solution: **Colab**{: .label .label-green }[Unit Sphere](https://colab.research.google.com/github/fum-cs/fds/blob/main/code/hw1/fatehinia_data_algo.ipynb)
* SAT
* SA, Continue
  : Chapter 3 of [My Book](https://www.dropbox.com/s/8bnxpzvfgiwma0k/combopt-PSO-20160514.pdf?dl=0)

**HW4**{: .label .label-red }[Python Code of Program 1.3, Page 23 of My book - SA](https://vu.um.ac.ir/mod/assign/view.php?id=449055&forceview=1), due: 1402/07/26

**Further Reading**{: .label .label-yellow }
* [Simulated Annealing From Scratch in Python](https://machinelearningmastery.com/simulated-annealing-from-scratch-in-python/)
* [Simulated Annealing Tutorial, 2D Example](http://apmonitor.com/me575/index.php/Main/SimulatedAnnealing)


## 1402/07/22
PSO
  : Chapter 6 of [My Book](https://www.dropbox.com/s/8bnxpzvfgiwma0k/combopt-PSO-20160514.pdf?dl=0)

**HW5**{: .label .label-red }Use one the Python packages to find the minimum of $$f(x)=3sin(x)+(0.1x-3)^2$$: [PSO for function 1.2](https://vu.um.ac.ir/mod/assign/view.php?id=449056&forceview=1), due: 1402/07/28

Some Python packages for PSO:
  * [Pymoo](https://pymoo.org/algorithms/soo/pso.html)
  * [PySwarms](https://pyswarms.readthedocs.io/en/latest/index.html)

**Further Reading**{: .label .label-yellow }
- [A Gentle Introduction to Particle Swarm Optimization](https://machinelearningmastery.com/a-gentle-introduction-to-particle-swarm-optimization/)

**Paper**{: .label .label-blue }: [A Fish School Clustering Algorithm: Applied to Student Sectioning Problem](https://www.dropbox.com/scl/fi/89p637l3ok3k7zobc1msp/A-Fish-School-Clustering-Algorithm-Applied-to-Student-Sectioning-Problem.pdf?rlkey=0540d3lwox88jdxx781swxd2q&dl=0)


## 1402/07/24

* Chapter 11 & 12 of Pattern Recognition, Theodoridis
* Chapeter 11:
  - Page 602, Section 11.2 PROXIMITY MEASURES - Page 604
  - Page 606, Section B. Similarity Measures: The inner product & Pearson’s correlation coefficient
  - Page 607, Discrete-Valued Vectors & contingency table
  - Page 616, 11.2.3 Proximity Functions between a Point and a Set

* Chapter 12:
  - 12.1 INTRODUCTION
  - 12.3 SEQUENTIAL CLUSTERING ALGORITHMS

## 1402/07/29
* **HW2-Sol**{: .label .label-green }, Solution: 
  - **Colab**{: .label .label-green }[Fatehinia](https://colab.research.google.com/github/fum-cs/fds/blob/main/code/hw2/Fatehinia.ipynb)
  - **Colab**{: .label .label-green }[Bagherpour](https://colab.research.google.com/github/fum-cs/fds/blob/main/code/hw2/Bagherpour.ipynb)

* Stirling Numbers, Recursive Functions & SAT Table
  - **Colab**{: .label .label-green }[SAT Table](https://colab.research.google.com/github/fum-cs/fds/blob/main/code/SAT_Table.ipynb)

## 1402/08/01

* SAT Table & Brute Force Algorithm for Clustering
  - **Colab**{: .label .label-green }[Brute Force Alg for Clustering](https://colab.research.google.com/github/fum-cs/fds/blob/main/code/BF-clustering.ipynb)
  
**HW6**{: .label .label-red }[Generate data & Clustering](https://vu.um.ac.ir/mod/assign/view.php?id=454178), due: 1402/08/04

**HW7**{: .label .label-red }[BSAS Algorithm](https://vu.um.ac.ir/mod/assign/view.php?id=454181), due: 1402/08/07

### Image Processing and Computer Vision, Intro

**Colab**{: .label .label-green }[Image Segmentation 01](https://colab.research.google.com/github/fum-cs/fds/blob/main/code/image_seg_01.ipynb)

  
## 1402/08/06

### Representative-Based Clustering

* Section 14.3.5 of [ESL](https://fumdrive.um.ac.ir/index.php/s/FH8nB4SwGkJrMeQ)
   - Page 527/764 ESL, Eq. 14.28: W(C)
   - The problem with one unknown variable becomes a problem with two unknowns!
* Section 8.3 of [K-means Clustering](https://www.dropbox.com/s/mrxdshg6nx98ojk/kmeans-clustering.pdf?dl=1)
* **Colab**{: .label .label-green }[Image Segmentation 02- kmeans clustering](https://colab.research.google.com/github/fum-cs/fds/blob/main/code/image_seg_02_k-means.ipynb)
* Chapter 13 of [Data Mining & Analysis](https://dataminingbook.info/)  
* [Slides (Representative-based Clustering)](https://www.cs.rpi.edu/~zaki/DMML/slides/pdf/ychap13.pdf)

**Further Reading**{: .label .label-yellow }

* [Lloyd’s, MacQueen’s and Hartigan-Wong’s k-Means](https://towardsdatascience.com/three-versions-of-k-means-cf939b65f4ea)
* [Convergence in Hartigan-Wong k-means method and other algorithms](https://datascience.stackexchange.com/questions/9858/convergence-in-hartigan-wong-k-means-method-and-other-algorithms)

### 1402/08/13

* [sklearn.datasets.make_blobs](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html)
* Section 14.3.6 of [ESL](https://fumdrive.um.ac.ir/index.php/s/FH8nB4SwGkJrMeQ)
   - Page 528/764 ESL, K-means

### 1402/08/15
* Section 14.3.9 of [ESL](https://fumdrive.um.ac.ir/index.php/s/FH8nB4SwGkJrMeQ)
   - Page 533/764 Vector Quantization

**Colab**{: .label .label-green }[Image Segmentation 03- kmeans clustering](https://colab.research.google.com/github/fum-cs/fds/blob/main/code/image_seg_03_k-means_centers.ipynb)

**Colab**{: .label .label-green }[LVQ](https://colab.research.google.com/github/fum-cs/fds/blob/main/code/LVQ.ipynb)

**HW8**{: .label .label-red }[K-means on color images](https://vu.um.ac.ir/mod/assign/view.php?id=458799), due: 1402/08/19

**Further Reading**{: .label .label-yellow }

* [Clustering, Lecture 14, New York University](https://people.csail.mit.edu/dsontag/courses/ml12/slides/lecture14.pdf) 
* [CSC 411 Lecture 15: K-Means, University of Toronto](https://www.cs.toronto.edu/~rgrosse/courses/csc411_f18/slides/lec15-slides.pdf)


## 1402/08/20,22

### High Dimensional Data

* [The curse of dimensionality, Candy example](../misc/Curse%20of%20Dimensionality%20-%20Candy-Example.pdf)

* Section 2.5 of [ESL](https://fumdrive.um.ac.ir/index.php/s/FH8nB4SwGkJrMeQ)
   - Page 41/764 Local Methods in High Dimensions

* Slides [Chap. 1 of Zaki](https://www.cs.rpi.edu/~zaki/DMML/slides/pdf/ychap1.pdf)

* Slides [Chap. 6 of Zaki](https://www.cs.rpi.edu/~zaki/DMML/slides/pdf/ychap6.pdf)

**Colab**{: .label .label-green }[High Dimensional Data - The curse of dimensionality](https://colab.research.google.com/github/fum-cs/fds/blob/main/code/high_dim_and_CD.ipynb)

**HW9**{: .label .label-red }[Page 15 of FDS - Orthogonality of d-dimensional Gaussian vectors](https://vu.um.ac.ir/mod/assign/view.php?id=460674), due: 1402/08/26

**Colab**{: .label .label-green }[High Dimensional Data - KNN](https://colab.research.google.com/github/fum-cs/fds/blob/main/code/high_dim_and_KNN.ipynb)

**Colab**{: .label .label-green }[Clustering of images, as high dim. data](https://colab.research.google.com/github/fum-cs/fds/blob/main/code/image_clustering.ipynb)

**Further Reading**{: .label .label-yellow }
* [Random Projection: Theory and Implementation in Python with Scikit-Learn](https://stackabuse.com/random-projection-theory-and-implementation-in-python-with-scikit-learn/)
* [Johnson–Lindenstrauss lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma)
* [Gaussian random projection](https://en.wikipedia.org/wiki/Random_projection)
* [Scikit-learn: The Johnson-Lindenstrauss bound for embedding with random projections](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_johnson_lindenstrauss_bound.html#sphx-glr-auto-examples-miscellaneous-plot-johnson-lindenstrauss-bound-py)

**Paper**{: .label .label-blue }[Supervised dimensionality reduction for big data](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8129083/)

**Paper**{: .label .label-blue }[An Introduction to Johnson–Lindenstrauss Transforms
](https://arxiv.org/pdf/2103.00564.pdf)
* [Sketching Algorithms for Big Data, Harvard](https://www.sketchingbigdata.org/fall17/)
  - [JL Lemma, History of lower bounds](https://www.sketchingbigdata.org/fall17/lec/lec5.pdf)
  - [JL Lower bound Optimality](https://www.sketchingbigdata.org/fall17/lec/lec6.pdf)

## 1402/09/04

### Bias-Variance Tradeoff
* [Wiki](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)
* [MLU-Explain bias-variance](https://mlu-explain.github.io/bias-variance/)
* [MLU-Explain double-descent, part 1](https://mlu-explain.github.io/double-descent/)

**Further Reading**{: .label .label-yellow }

* [MLU-Explain double-descent, part 2](https://mlu-explain.github.io/double-descent2/)
* [The Bias-Variance Tradeoff: A Newbie’s Guide, by a Newbie](https://medium.com/@DeepthiTabithaBennet/the-bias-variance-tradeoff-a-newbies-guide-by-a-newbie-95fb03dbebcb)
* [bias-variance-trade-off](https://spotintelligence.com/2023/04/11/bias-variance-trade-off/)

**Paper**{: .label .label-blue }[VC Theoretical Explanation of Double Descent](https://arxiv.org/abs/2205.15549)

**Paper**{: .label .label-blue }[Reconciling modern machine-learning practice and the classical bias–variance trade-off](https://www.pnas.org/doi/10.1073/pnas.1903070116)
  - [Double Descent](https://medium.com/mlearning-ai/double-descent-8f92dfdc442f), [Highlights](misc/medium-com_mlearning-ai_double-descent-highlightes.md)
  - [Reproducing Deep Double Descent](https://hippocampus-garden.com/double_descent/), [Highlights](misc/hippocampus-garden-com_double_descent-highlightes.md)
    + [deep_double_descent, colab](https://colab.research.google.com/drive/1lT2dUqal90NbLVQIGvseyAdKzH19MH2T?usp=sharing)
* [Sec 22.3 of Zaki](https://fumdrive.um.ac.ir/index.php/f/4160875)

**Paper**{: .label .label-blue }[Understanding the double descent curve in Machine Learning](https://arxiv.org/abs/2211.10322)

## 1402/09/06,11,13

* [Chapter 5 of VanderPlas: In Depth: k-Means Clustering](https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html)
* Chapter 17 of [Data Mining & Analysis](https://dataminingbook.info/)  
* [Clustering Validation](https://www.cs.rpi.edu/~zaki/DMML/slides/pdf/ychap17.pdf)
* [Matching in Bipartite Graphs](https://math.libretexts.org/Courses/Saint_Mary%27s_College_Notre_Dame_IN/SMC%3A_MATH_339_-_Discrete_Mathematics_(Rohatgi)/Text/5%3A_Graph_Theory/5.6%3A_Matching_in_Bipartite_Graphs)
* [Silhouette, Clustering Evaluation](https://en.wikipedia.org/wiki/Silhouette_(clustering))
* [Clustering Evaluations](misc/Clustering-Evaluation-Measures.md)

**Colab**{: .label .label-green }[bi-partiate-graph-maximum-matching](https://colab.research.google.com/github/mamintoosi/DM/blob/master/code/bi-partiate-graph-maximum-matching.ipynb)

**Colab**{: .label .label-green }[Silhouette](https://colab.research.google.com/github/fum-cs/fds/blob/main/code/Silhouette-clustering.ipynb)

**Further Reading**{: .label .label-yellow }
* [MSc. Project: Graph Cut](http://hcloud.hsu.ac.ir/index.php/s/OtehB4LUQFv8cla)
* [Wiki: Graph Matching](https://fa.wikipedia.org/wiki/%D8%AA%D8%B7%D8%A7%D8%A8%D9%82_(%DA%AF%D8%B1%D8%A7%D9%81))
* [Bipartite Graphs and Stable Matchings](https://mathbooks.unl.edu/Contemporary/sec-graph-bipartite.html)
* [MIT, NOTES ON MATCHING](https://math.mit.edu/~djk/18.310/Lecture-Notes/MatchingProblem.pdf)

**Paper**{: .label .label-blue }[Graph Matching and local search](https://www.dropbox.com/s/y9cb52otfdnrsin/1396-AIMC48-Ezzati-Graph%20Matching%20and%20Stochastic%20Search.pdf?dl=11)

**Paper**{: .label .label-blue }[Graph Feature Selection for Anti-Cancer Plant Recommendation](https://mathco.journals.pnu.ac.ir/article_10067.html)


## 1402/09/18, 25, 27

* [Principal Component Analysis explained visually](https://setosa.io/ev/principal-component-analysis/)
* [In Depth: Principal Component Analysis, Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html)
* [PRML-PCA Slides](https://www.dropbox.com/s/ftw3f8wvx07gos9/PRML_ch12_sec1.pdf?dl=0)
  - [Matrix Differentiation](misc/MatrixCalculus.pdf) by 
Randal J. Barnes
* [Chapter 7 of Zaki (Slides)](https://www.cs.rpi.edu/~zaki/DMML/slides/pdf/ychap7.pdf)

**Colab**{: .label .label-green }[PCA-01](https://colab.research.google.com/github/fum-cs/fds/blob/main/code/PCA/PCA_01.ipynb)

**Kaggle**{: .label .label-green }[Country Profiling Using PCA and Clustering](https://www.kaggle.com/leo2510/country-clustering-and-pca-analysis)

* [An Introduction to Principal Component Analysis (PCA) with 2018 World Soccer Players Data](https://blog.exploratory.io/an-introduction-to-principal-component-analysis-pca-with-2018-world-soccer-players-data-810d84a14eab), [PDF](../misc/An%20Introduction%20to%20Principal%20Component%20Analysis%20(PCA)%20with%202018%20World%20Soccer%20Players%20Data.pdf)
* [Using PCA to See Which Countries have Better Players for World Cup Games](https://blog.exploratory.io/using-pca-to-see-which-countries-have-better-players-for-world-cup-games-a72f91698b95), [PDF](../misc/Using%20PCA%20to%20See%20Which%20Countries%20have%20Better%20Players%20for%20World%20Cup%20Games.pdf)

**HW10**{: .label .label-red }[PCA Algorithm](https://vu.um.ac.ir/mod/assign/view.php?id=468055), due: 1402/10/02

**Further Reading**{: .label .label-yellow }

* [A geometric interpretation of the covariance matrix](https://www.visiondummy.com/2014/04/geometric-interpretation-covariance-matrix/)
* [A geometric interpretation of ... (In Persian)](https://www.dropbox.com/s/pqnemuis96vebuu/main.pdf?dl=0)
* [PCA in SKLearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
* [PCA on IRIS](https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html)
* [Faces recognition example using eigenfaces and SVMs](https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html#sphx-glr-auto-examples-applications-plot-face-recognition-py)
    - [Face dataset](http://vis-www.cs.umass.edu/lfw/alpha_all_35.html)

**Paper**{: .label .label-blue }[Eigenbackground Revisited](https://github.com/mamintoosi/Eigenbackground-Revisited)

**Colab**{: .label .label-green }[SVD-01](https://colab.research.google.com/github/fum-cs/fds/blob/main/code/SVD/SVD_01.ipynb)

**Colab**{: .label .label-green }[SVD for Image Compression](https://colab.research.google.com/github/fum-cs/fds/blob/main/code/SVD/svd_image_compression.ipynb)


## 1402/10/02
## Hierarchical Clustering
* [A good image for hierarchical clustering](https://www.analyticsvidhya.com/blog/2021/06/single-link-hierarchical-clustering-clearly-explained/)
* Chapter 14 of [Data Mining & Analysis](https://dataminingbook.info/)  
* Slides (Hierarchical Clustering): [PDF](https://www.cs.rpi.edu/~zaki/DMML/slides/pdf/ychap14.pdf)
* [sklearn.cluster.AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering)
  - [Comparing different hierarchical linkage methods on toy datasets](https://scikit-learn.org/stable/auto_examples/cluster/plot_linkage_comparison.html)

**Colab**{: .label .label-green }[Clustering of images](https://colab.research.google.com/github/fum-cs/fds/blob/main/code/image-clustering_HC.ipynb)


**Further Reading**{: .label .label-yellow }
    
- Slide: [Hierarchical Clustering](https://cse.buffalo.edu/~jing/cse601/fa12/materials/clustering_hierarchical.pdf) by Jing Gao 

## 1402/10/04
## Linear Discriminant Analysis
* Chapter 20 of [Data Mining & Analysis](https://dataminingbook.info/)  
* Slides (Linear Discriminant Analysis): [PDF](https://www.cs.rpi.edu/~zaki/DMML/slides/pdf/ychap20.pdf)
* [Comparison of LDA and PCA](https://scikit-learn.org/0.16/auto_examples/decomposition/plot_pca_vs_lda.html#example-decomposition-plot-pca-vs-lda-py)
* HW: Compare LDA and PCA first axis (classification by SVM)

## 1402/10/09

**Mid Term**{: .label .label-purple }


## 1402/10/11

* [Soft k-means](../misc/soft-k-means.pdf)

**Colab**{: .label .label-green } [Gaussian Mixture Models](https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.12-Gaussian-Mixtures.ipynb)

## 1402/10/16,18
## Bayes

* [Slides](https://www.dropbox.com/s/cv22u3jlx2lrre5/Bayesian%20Classification%20withInsect_examples.pdf?dl=0)
* [SE example](https://stackoverflow.com/questions/10059594/a-simple-explanation-of-naive-bayes-classification)
* [How to Develop a Naive Bayes Classifier from Scratch in Python](https://machinelearningmastery.com/classification-as-conditional-probability-and-the-naive-bayes-algorithm/)


**Further Reading**{: .label .label-yellow }

* [Duda](https://www.dropbox.com/s/d5ordbb9kf37398/Bayesian.pdf?dl=0)
* [Naïve Bayes Algorithm -Implementation from scratch in Python, Medium](https://medium.com/@rangavamsi5/na%C3%AFve-bayes-algorithm-implementation-from-scratch-in-python-7b2cc39268b9)
  - [Github](https://github.com/vamc-stash/Naive-Bayes)
* [Segmentation using Bayesian Decision Theory](https://www.cs.utexas.edu/~grauman/courses/fall2008/slides/segmentation_bayes_aggarwal.pdf)

**Paper**{: .label .label-blue }[BayeSeg: Bayesian modeling for medical image segmentation with interpretable generalizability](https://www.sciencedirect.com/science/article/pii/S1361841523001494)
  - [Github](https://github.com/obiyoag/BayeSeg)

* [Bayesian approach to Natural Image Matting](https://github.com/praveenVnktsh/Bayesian-Matting)



## 1402 Winter
### Adding Features

**Colab**{: .label .label-green }[Add Pixels' coordinates for image segmentation](https://colab.research.google.com/github/fum-cs/fds/blob/main/code/image_seg_04_coords.ipynb)

## 1402/11/01

**EXAM**{: .label .label-purple }

### Image Processing and Computer Vision
- Website: [Image Processing in Python with Scikit-image](https://blog.faradars.org/image-processing-in-python/) by M. Jaderian 
  * [Scikit-image documentation](https://scikit-image.org/docs/stable/)
  * [Scikit-image examples](https://scikit-image.org/docs/stable/auto_examples/index.html)
- Website: [Image Processing in Python with OpenCV](https://www.m-vision.ir/%D8%A2%D9%85%D9%88%D8%B2%D8%B4/%D9%BE%D8%B1%D8%AF%D8%A7%D8%B2%D8%B4-%D8%AA%D8%B5%D9%88%DB%8C%D8%B1/opencv/%D8%A2%D9%85%D9%88%D8%B2%D8%B4-%D9%BE%D8%B1%D8%AF%D8%A7%D8%B2%D8%B4-%D8%AA%D8%B5%D9%88%DB%8C%D8%B1-%D8%A8%D8%A7-%D9%BE%D8%A7%DB%8C%D8%AA%D9%88%D9%86-%D8%AA%D9%88%D8%B3%D8%B7-opencv/) by M. Kiani 
- Github: [Tutorial for Image Processing in Python](https://github.com/zengsn/image-processing-python) by Shaoning Zeng 
- Book: [Image processing tutorials](https://github.com/yg42/iptutorials/blob/master/book/tutorials_python.pdf)
    

**Further Reading**{: .label .label-yellow }
: [Some published papers](https://fumcs.github.io/projects/computer-vision/)
* [الگوریتم مجارستانی - ویکی‌پدیا، دانشنامهٔ آزاد (wikipedia.org)](https://fa.wikipedia.org/wiki/%D8%A7%D9%84%DA%AF%D9%88%D8%B1%DB%8C%D8%AA%D9%85_%D9%85%D8%AC%D8%A7%D8%B1%D8%B3%D8%AA%D8%A7%D9%86%DB%8C)

### Image Matting
- Github Rep.: [A Python library for alpha matting](https://github.com/pymatting/pymatting) [https://pymatting.github.io/](https://pymatting.github.io/) by Y. Gavet & J. Debayle  


## K-means

* Sec 5.11 of [JakeVanderPlas](https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html)

**Paper**{: .label .label-blue } [كلاسه بندی فازی بهینه دانشجویان با استفاده از یك تابع فازی در حل مسئله برنامه ریزی ژنتیكی دروس هفتگی دانشگاه](https://www.dropbox.com/s/7ohrff9c6im8bye/1382-StudentSectioning.pdf?dl=1)

* [Bilateral K-Means for Superpixel Computation](https://hal-enpc.archives-ouvertes.fr/hal-03651336/document)
* [Balanced clustering - Wikipedia](https://en.wikipedia.org/wiki/Balanced_clustering) 
* [balanced-kmeans · PyPI](https://pypi.org/project/balanced-kmeans/)
* [K-means using PyTorch (github.com)](https://github.com/subhadarship/kmeans_pytorch)
* [Balanced K-Means for Clustering](https://link.springer.com/chapter/10.1007/978-3-662-44415-3_4)
* [Balanced k-Means Revisited](https://openreview.net/forum?id=VndsvZYlMo&amp;referrer=%5BTMLR%5D(%2Fgroup%3Fid%3DTMLR))
* [K-Means Clustering in Python: A Practical Guide – Real Python](https://realpython.com/k-means-clustering-python/)
* [Data clustering: 50 years beyond K-means - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167865509002323)
* [K-Means Factorization](https://www.dropbox.com/s/2fsmpj7i7x3rngy/K-Means%20Factorizaton1512.07548.pdf?dl=1)
* [Clustering IRIS dataset with particle swarm optimization(PSO)](https://github.com/NiloofarShahbaz/PSOClustering)



### Representative-Based Clustering

- Chapter 13 of [Data Mining & Analysis](https://dataminingbook.info/)  
- **HW**{: .label .label-red } 13.5: Q2, Q4, Q6, Q7 
- Slides (Representative-based Clustering): [PDF](https://www.cs.rpi.edu/~zaki/DMML/slides/pdf/ychap13.pdf)
- Slide: [Introduction to Machine Learning (Clustering and EM)](http://www.cs.cmu.edu/~aarti/Class/10701_Spring14/slides/EM.pdf) by Barnabás Póczos & Aarti Singh 
- Tutorial: [The Expectation Maximization Algorithm](https://www.cs.utah.edu/~piyush/teaching/EM_algorithm.pdf) by 
Sean Borman 
- Tutorial: [What is Bayesian Statistics?](http://www.bandolier.org.uk/painres/download/whatis/What_is_Bay_stats.pdf) by John W Stevens
    
**Further Reading**{: .label .label-yellow }
    
- Slide: [Tutorial on Estimation and Multivariate Gaussians](http://ttic.uchicago.edu/~shubhendu/Slides/Estimation.pdf) by Shubhendu Trivedi 
- Slide: [Mixture Model](https://cse.buffalo.edu/~jing/cse601/fa12/materials/clustering_mixture.pdf) by Jing Gao 
- Paper: [Fast Exact k-Means, k-Medians and Bregman Divergence Clustering in 1D](https://cs.au.dk/~larsen/papers/1dkmeans.pdf) 
- Paper: [k-Means Requires Exponentially Many Iterations Even in the Plane](http://cseweb.ucsd.edu/~avattani/papers/kmeans.pdf) by Andrea Vattani 
- Book: [Understanding Machine Learning: From Theory to Algorithms](https://www.amazon.com/Understanding-Machine-Learning-Theory-Algorithms/dp/1107057132) by Shai Shalev-Shwartz and Shai Ben-David 

## Mahalanobis distance

* Chapter 2 of Zaki, page 54, eq. 2.43
* [What is Mahalanobis distance?](https://blogs.sas.com/content/iml/2012/02/15/what-is-mahalanobis-distance.html)
  - [Use the Cholesky transformation to correlate and uncorrelate variables](https://blogs.sas.com/content/iml/2012/02/08/use-the-cholesky-transformation-to-correlate-and-uncorrelate-variables.html)
* [Mahalanobis Distance – Understanding the math with examples (python)](https://www.machinelearningplus.com/statistics/mahalanobis-distance/)
* [Unlocking the Power of Mahalanobis Distance: Exploring Multivariate Data Analysis with Python](https://medium.com/@the_daft_introvert/mahalanobis-distance-5c11a757b099)
* [Outlier detection-faradars](https://blog.faradars.org/%D8%AF%D8%A7%D8%AF%D9%87%E2%80%8C-%D9%BE%D8%B1%D8%AA-%D8%A8%D8%A7-%D9%81%D8%A7%D8%B5%D9%84%D9%87-%D9%85%D8%A7%D9%87%D8%A7%D9%84%D8%A7%D9%86%D9%88%D8%A8%DB%8C%D8%B3/)
* [Mahalanobis Distance – Understanding the math with examples (python)](https://www.machinelearningplus.com/statistics/mahalanobis-distance/)


## 1402 Winter
### <a name="L12"></a>Clustering Validation
- Chapter 17 of [Data Mining & Analysis](https://dataminingbook.info/)  
- Slides of Section 17.1 (Clustering Validation): [PDF](https://www.cs.rpi.edu/~zaki/DMML/slides/pdf/ychap17.pdf)
- Slide: [Clustering Analysis](http://www.mind.disco.unimib.it/public/opere/143.pdf) by Enza Messina 
- Slide: [Information Theory](http://www2.eng.cam.ac.uk/~js851/3F1MM13/L1.pdf) by Jossy Sayir 
- Slide: [Normalized Mutual Information: Estimating Clustering Quality](https://course.ccs.neu.edu/cs6140sp15/7_locality_cluster/Assignment-6/NMI.pdf) by Bilal Ahmed  
    
**Further Reading**{: .label .label-yellow }
    
- Slide: [Clustering Evaluation (II)](http://eniac.cs.qc.cuny.edu/andrew/gcml/lecture23.pdf) by Andrew Rosenberg 
- Slide: [Evaluation (I)](http://eniac.cs.qc.cuny.edu/andrew/gcml/lecture22.pdf) by Andrew Rosenberg 

## 1402 Winter
### <a name="L10"></a>Density-Based Clustering
- Chapter 15 of [Data Mining & Analysis](https://dataminingbook.info/)  
- Slides of Section 15.1 (Density-based Clustering): [PDF](https://www.cs.rpi.edu/~zaki/DMML/slides/pdf/ychap15.pdf)
- Slide: [Spatial Database Systems](http://dna.fernuni-hagen.de/Tutorial-neu.pdf) by 
Ralf Hartmut Güting 

## 1402 Winter
### <a name="L11"></a>Kernel Method
    
- Chapter 5 of [Data Mining & Analysis](https://dataminingbook.info/)  
- Kernel-Kmeans Chapter 13 of [Data Mining & Analysis](https://dataminingbook.info/) 
- **HW**{: .label .label-red } TBA

**EXAM**{: .label .label-purple }

## 1402 Winter
### <a name="L11"></a>Spectral and Graph Clustering  
- Chapter 16 of [Data Mining & Analysis](https://dataminingbook.info/)  
    **Exercises** 16.5: Q2, Q3, Q6 
- Slides (Spectral and Graph Clustering): [PDF](https://www.cs.rpi.edu/~zaki/DMML/slides/pdf/ychap16.pdf) 
- Slide: [Spectral Clustering](http://eniac.cs.qc.cuny.edu/andrew/gcml/lecture21.pdf) by Andrew Rosenberg 
- Slide: [Introduction to Spectral Clustering](http://www.cvl.isy.liu.se:82/education/graduate/spectral-clustering/SC_course_part1.pdf) by Vasileios Zografos and Klas Nordberg 
    
**Further Reading**{: .label .label-yellow }
    
- Slide: [Spectral Methods](https://cse.buffalo.edu/~jing/cse601/fa12/materials/clustering_spectral.pdf) by Jing Gao 
- Tutorial: [A Tutorial on Spectral Clustering](https://arxiv.org/pdf/0711.0189.pdf) by Ulrike von Luxburg 
- Tutorial: [Matrix Differentiation](https://atmos.washington.edu/~dennis/MatrixCalculus.pdf) by 
Randal J. Barnes
- Lecture: [Spectral Methods](https://cseweb.ucsd.edu/~dasgupta/291-unsup/lec7.pdf) by Sanjoy Dasgupta 
- Paper: [Positive Semidefinite Matrices and Variational Characterizations of Eigenvalues
](http://www.ee.cuhk.edu.hk/~wkma/engg5781/new_notes/lecture%204-%20PSD-%20note.pdf) by Wing-Kin Ma  

### <a name="L7"></a>Itemset Mining
- Chapter 8 of [Data Mining & Analysis](https://dataminingbook.info/)  

## 1402 Winter
### <a name="L6"></a>Link Analysis         
- Ranking Graph Vertices, [Page Rank](http://mct.iranjournals.ir/article_263_d3eae82ca520e66df01732cb07fd6841.pdf)
- [Linear Algebra and Technology](https://tmsj.um.ac.ir/article_42621_7382212ff95b4771e39264078f80ba37.pdf)

**Further Reading**{: .label .label-yellow }

- Chapter 5 of [Mining of Massive Datasets](http://www.mmds.org)  
- Slide of Sections 5.1, 5.2 (PageRank, Efficient Computation of PageRank): [Analysis of Large Graphs 1](http://www.mmds.org/mmds/v2.1/ch05-linkanalysis1.pdf)
- Slide of Sections 5.3-5.5 (Topic-Sensitive PageRank, Link Spam, Hubs and Authorities): [Analysis of Large Graphs 2](http://www.mmds.org/mmds/v2.1/ch05-linkanalysis1.pdf)
- Slide: [The Linear Algebra Aspects of PageRank](http://www4.ncsu.edu/~ipsen/ps/slides_dagstuhl07071.pdf) by Ilse Ipsen     
- Paper: [A Survey on Proximity Measures for Social Networks](https://link.springer.com/chapter/10.1007/978-3-642-34213-4_13) by Sara Cohen, Benny Kimelfeld, Georgia Koutrika 

### <a name="Additional-Slides"></a>Additional Slides:
* [Practical Data Science](http://www.datasciencecourse.org/lectures/) by Zico Kolter
* [Course: Data Mining](https://datalab.snu.ac.kr/~ukang/courses/18S-DM/) by U Kang
* [Statistical Data Mining Tutorials](http://www.cs.cmu.edu/~./awm/tutorials/index.html) by  Andrew W. Moore  

- Lecture: [Finding Meaningful Clusters in Data](https://cseweb.ucsd.edu/~dasgupta/291-unsup/lec5.pdf) by Sanjoy Dasgupta 
- Paper: [An Impossibility Theorem for Clustering](https://www.cs.cornell.edu/home/kleinber/nips15.pdf) by Jon Kleinberg 
