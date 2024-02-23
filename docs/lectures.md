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

## 1402/12

## Regression & Neural Networks

* See [this Jupyter-book](https://fum-cs.github.io/neural-networks) about Neural Networks