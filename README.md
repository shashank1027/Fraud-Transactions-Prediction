# Fraud Detection using Isolation Forest (AWS SageMaker)

This project trains an Isolation Forest model in AWS SageMaker using parquet files stored in Amazon S3.

### ✅ Tech Stack
- Python
- AWS SageMaker
- S3
- Scikit-Learn
- PyArrow

### 🚀 Steps
1. Data is cleaned into parquet files and stored in S3.
2. `train.py` is sent to SageMaker to run training.
3. SageMaker creates a training job and saves the model artifact in S3.
4. (optional) Deploy as real-time endpoint or batch inference job.

### 📂 Project Files
| File | Description |
|------|-------------|
| `train.py` | Contains model training script executed by SageMaker container |
| `requirements.txt` | Dependencies installed in the training container |
| `train.ipynb` | Notebook that triggers `estimator.fit()` |

### 🔧 Next steps (not yet implemented)
- Batch inference using SageMaker Batch Transform
- Real-time endpoint deployment


