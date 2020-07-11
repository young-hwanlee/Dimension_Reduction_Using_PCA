# Dimensionality Reduction Using Principal Component Analysis (PCA)
An implementation of principal component analysys

## Dataset
The face images are obtained from the software-based aligned version in [1], of the Labeled Faces in the Wild (LFW) database [2]. N = 1924 have been selected and cropped in order to omit most of the background.

## Results
### 1. Sigular Values
Sigular values are plotted. It seems that K=200~500 is enough to represent all faces.
![singular_values](https://user-images.githubusercontent.com/67979833/87225474-4973fd80-c35b-11ea-97e8-900191778137.png)

### 2.
Only 10 eigenfaces are shown.
![eigenfaces](https://user-images.githubusercontent.com/67979833/87225473-4973fd80-c35b-11ea-9744-7dd4da1dbeb5.png)

### 3.
5 faces are dispalyed with original dimension (N = 1924) as well as lower dimension (K=50, 100, and 500). It seems that when K is small, it cannot reconstruct the original faces, but if K is increasing, it can represent the original faces well.
![dimensionality_reduction](https://user-images.githubusercontent.com/67979833/87225472-4842d080-c35b-11ea-9152-e07b3252605d.png)

## References
[1] L. Wolf, T. Hassner, Y. Taigman, Effective unconstrained face recognition by combining multiple descriptors and learned background statistics, IEEE Trans. Pattern Anal. Mach. Intell. 33 (10) (2011) 1978–1990.  
[2] E. Learned-Miller, G. Huang, A. RoyChowdhury, H. Li, and G. Hua, “Labeled Faces in the Wild: Updates and New Reporting Procedures,” Technical Report, University of Massachusetts, Amherst, No. UM-CS-2014- 003.  
