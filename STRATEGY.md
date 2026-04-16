## Scoring Strategies

### Top-k Mean (Best Practical Default)
- **How it works:**  
  For each CV, sort chunk scores in descending order and average only the top k scores (e.g., 3–8).
- **Benefit:**  
  Captures the idea that a CV should have multiple, strong areas of evidence without being diluted by unrelated sections.

### Weighted Top-k Mean
- **How it works:**  
  Similar to top-k mean, but assigns higher weights to higher-ranked chunks (e.g., 1.0, 0.8, 0.6, ...).
- **Benefit:**  
  Emphasizes the strongest evidence while still considering other relevant parts of the CV.

### Softmax Pooling (Smooth Max)
- **Formula:**  
  ```
  cv_score = log(sum(exp(alpha * s_i))) / alpha
  ```
  Where `alpha` controls sharpness:
    - Higher alpha ≈ max pooling
    - Lower alpha ≈ mean pooling
- **Benefit:**  
  Provides a smooth balance between “best chunk matters” and “multiple chunks matter”.

### Hybrid Score (Very Robust)
- **Formula:**  
  ```
  cv_score = w * max_score + (1-w) * topk_mean
  ```
  Example: `w = 0.4`, `k = 5`.
- **Benefit:**  
  Ensures that one very strong match positively influences the score, but overall consistency still matters.