---
layout: default
title: Template Matching
nav_order: 2
---

#  Template Matching

{: .no_toc }

## Table of contents

{: .no_toc .text-delta }

1. TOC
{:toc}

---

# Homework Assignment: Template Matching and Image Analysis

## Objective
The objective of this homework assignment is to introduce students to **template matching** using the **sum of squared differences** method. Students will apply this technique to matrices and then extend it to grayscale images.

## Tasks

### Task 1: Matrix Template Matching
1. **Matrix Preparation**:
   - Create two matrices, `image_matrix` and `template_matrix`, with the following values:
     ```python
     img = np.array([
         [[0], [1], [2], [10]],
         [[3], [4], [5], [11]],
         [[6], [7], [8], [12]],
     ])

     t = np.array([ 
         [[4], [5]],
         [[7], [12]],
     ])
     ```
   - The dimensions of `image_matrix` should be larger than `template_matrix`.

2. **Implement `sum_of_squared_differences`**:
   - Write a Python function called `sum_of_squared_differences(image, template)` that computes the sum of squared differences between the `image` and the `template`.
   - You can use the `sum_of_squared_differences` function from the [Template-Matching.ipynb](https://github.com/Sri-Sai-Charan/Template-Matching) notebook.

3. **Find the Best Match**:
   - Use the `sum_of_squared_differences` function to find the location of the best match (minimum sum of squared differences) within the `image_matrix`.
   - Display the matched region in the `image_matrix`.

### Task 2: Image Template Matching
1. **Image Preparation**:
   - Use the coin image from [here](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_template.html) (or use any other image of your choice).
   - Convert the image to grayscale.

2. **Apply Template Matching**:
   - Use the `sum_of_squared_differences` function to find the best match for the provided `template_matrix` within the larger image.
   - Modify the ` template_matching` function of the [mentioned notebook](https://github.com/Sri-Sai-Charan/Template-Matching), so that runs without CV2 library.
   - Display the original image, the template, and the matched region.

### Task 3: Extensions (Optional, without any grade)
1. **Scale and Rotation Invariance**:
   - Discuss how template matching can be extended to handle scale and rotation variations.
   - Explore other methods (e.g., [normalized cross-correlation](https://xcdskd.readthedocs.io/en/latest/cross_correlation/cross_correlation_coefficient.html)) that are more robust to such variations.

2. **Performance Optimization**:
   - Investigate ways to optimize the template matching process (e.g., using [integral images](https://levelup.gitconnected.com/the-integral-image-4df3df5dce35) or FFT-based methods).

3. Experiment with different templates and images.

4. Discuss the limitations of template matching (e.g., sensitivity to lighting changes, occlusions, and noise).