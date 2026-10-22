| scheme        | system                     | task    |    n |   accuracy | 95% CI      |   balanced acc |   macro F1 |
|:--------------|:---------------------------|:--------|-----:|-----------:|:------------|---------------:|-----------:|
| leaky-2023    | smallcnn-gpu-leaky-2023    | species | 3669 |      1     | 0.999-1.000 |          1     |      1     |
| leaky-2023    | smallcnn-leaky-2023        | species | 3669 |      0.987 | 0.983-0.990 |          0.988 |      0.985 |
| official      | classic-hog-logreg         | species | 3669 |      0.702 | 0.686-0.717 |          0.688 |      0.676 |
| official      | resnet18-official          | species | 3669 |      0.972 | 0.967-0.977 |          0.975 |      0.968 |
| official      | smallcnn-gpu-official      | species | 3669 |      0.952 | 0.945-0.959 |          0.948 |      0.945 |
| official      | smallcnn-official          | species | 3669 |      0.915 | 0.905-0.923 |          0.912 |      0.904 |
| unseen-breeds | classic-hog-logreg         | species | 2395 |      0.649 | 0.628-0.666 |          0.641 |      0.628 |
| unseen-breeds | resnet18-unseen-breeds     | species | 2395 |      0.94  | 0.930-0.949 |          0.927 |      0.932 |
| unseen-breeds | smallcnn-gpu-unseen-breeds | species | 2395 |      0.895 | 0.883-0.908 |          0.875 |      0.88  |
| unseen-breeds | smallcnn-unseen-breeds     | species | 2395 |      0.875 | 0.861-0.888 |          0.872 |      0.863 |
