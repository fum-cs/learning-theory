---
layout: page
title: Algorithms for Data Science 
description: Listing of course modules and topics.
mathjax: true
nav_order: 2
---

## 1402/11/23

### Image Processing and Computer Vision
- Website: [Image Processing in Python with Scikit-image](https://blog.faradars.org/image-processing-in-python/) by M. Jaderian 
  * [Scikit-image documentation](https://scikit-image.org/docs/stable/)
  * [Scikit-image examples](https://scikit-image.org/docs/stable/auto_examples/index.html)
- Website: [Image Processing in Python with OpenCV](https://www.m-vision.ir/%D8%A2%D9%85%D9%88%D8%B2%D8%B4/%D9%BE%D8%B1%D8%AF%D8%A7%D8%B2%D8%B4-%D8%AA%D8%B5%D9%88%DB%8C%D8%B1/opencv/%D8%A2%D9%85%D9%88%D8%B2%D8%B4-%D9%BE%D8%B1%D8%AF%D8%A7%D8%B2%D8%B4-%D8%AA%D8%B5%D9%88%DB%8C%D8%B1-%D8%A8%D8%A7-%D9%BE%D8%A7%DB%8C%D8%AA%D9%88%D9%86-%D8%AA%D9%88%D8%B3%D8%B7-opencv/) by M. Kiani 
- Github: [Tutorial for Image Processing in Python](https://github.com/zengsn/image-processing-python) by Shaoning Zeng 
- Book: [Image processing tutorials](https://github.com/yg42/iptutorials/blob/master/book/tutorials_python.pdf)
    
**HW1**{: .label .label-red }[Template Matching from Scrach](./hws/Template-Matching), due date: 1402/11/27


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
