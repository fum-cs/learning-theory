---
layout: page
title: Lectures
description: Listing of course modules and topics.
mathjax: true
nav_order: 2
---

<div>
{% assign instructors = site.staffers | where: 'role', 'Instructor' %}
  <div class="role">
    {% for staffer in instructors %}
    {{ staffer }}
    {% endfor %}
  </div>
</div>

{: .highlight }
> Welcome to the Course!


## 1402/11/23

### Image Data
- Website: [Image Processing in Python with Scikit-image](https://blog.faradars.org/image-processing-in-python/) by M. Jaderian 
  * [Scikit-image documentation](https://scikit-image.org/docs/stable/)
  * [Scikit-image examples](https://scikit-image.org/docs/stable/auto_examples/index.html)
- Website: [Image Processing in Python with OpenCV](https://www.m-vision.ir/%D8%A2%D9%85%D9%88%D8%B2%D8%B4/%D9%BE%D8%B1%D8%AF%D8%A7%D8%B2%D8%B4-%D8%AA%D8%B5%D9%88%DB%8C%D8%B1/opencv/%D8%A2%D9%85%D9%88%D8%B2%D8%B4-%D9%BE%D8%B1%D8%AF%D8%A7%D8%B2%D8%B4-%D8%AA%D8%B5%D9%88%DB%8C%D8%B1-%D8%A8%D8%A7-%D9%BE%D8%A7%DB%8C%D8%AA%D9%88%D9%86-%D8%AA%D9%88%D8%B3%D8%B7-opencv/) by M. Kiani 
- Github: [Tutorial for Image Processing in Python](https://github.com/zengsn/image-processing-python) by Shaoning Zeng 
- Book: [Image processing tutorials](https://github.com/yg42/iptutorials/blob/master/book/tutorials_python.pdf)
    
**HW1**{: .label .label-red }[Template Matching from Scrach](./hws/Template-Matching), due date: 1402/12/03

**Further Reading**{: .label .label-yellow }

[Some published papers](https://fumcs.github.io/projects/computer-vision/)

## GitHub Pages

**HW2**{: .label .label-red }[Create your own GitHub page](./hws/GitHub-Pages), due date: 1402/12/04

# 1402/12

## Regression & Neural Networks

The initial section of the course focuses on **Neural Networks**, which can be accessed through [this Jupyter-book](https://fum-cs.github.io/neural-networks) or the [PDF of the book](https://fumdrive.um.ac.ir/index.php/s/ic2s8AGyFSZox8r).

## Regularization

* [Regularization in Machine Learning](https://www.geeksforgeeks.org/regularization-in-machine-learning/)
* [Regularization in Machine Learning (with Code Examples)](https://www.dataquest.io/blog/regularization-in-machine-learning/)
* [Logistic regression with L2 regularization for binary classification](https://github.com/pickus91/Logistic-Regression-Classifier-with-L2-Regularization)

# 1403 

## Linear Models, Kernel Methods, Bayesian and Ensemble Learning

The other section of the course focuses on other **Learning Concepts**, which can be accessed through [this Jupyter-book](https://fum-cs.github.io/machine-learning).




<!-- **HW4**{: .label .label-red }[Linear regression](https://ml-lectures.org/docs/supervised_learning_wo_NNs/Linear-regression.html), due date: 1402/12/20

**HW5**{: .label .label-red }[Linear regression, Regularization, Lasso](https://ml-course.github.io/master/labs/Lab%201a%20-%20Linear%20Models%20for%20Regression.html), due date: 1403/01/14 -->

## Genetic Algorithm

- [GA Slides](https://www.dropbox.com/scl/fi/dr2y0k2zgvy5ypyub3d2u/Genetic_Algorithms.pdf?rlkey=dtw9f2f2xnnkckk0bjbnswk89&dl=0)
- [PyGAD: Python library for building the genetic algorithm](https://pygad.readthedocs.io/en/latest/)
  - **Colab**{: .label .label-green }[PyGAD Sample](https://colab.research.google.com/github/fum-cs/learning-theory/blob/main/code/PyGAD-Sample.ipynb)

- GA Book: [Genetic Algorithms with Python](https://1drv.ms/b/s!AmjylFwPahYxgZYlVQRNrdSnez5J2A?e=pVh82p)
  - [Github](https://github.com/handcraftsman/GeneticAlgorithmsWithPython)
  - **Colab**{: .label .label-green }[A Sample without unittest](https://colab.research.google.com/github/fum-cs/learning-theory/blob/main/code/wo-unittest.ipynb)
