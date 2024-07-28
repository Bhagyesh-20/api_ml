# Crop Disease Prediction API

This repository contains an API that leverages a TensorFlow model to predict up to 20 different crop diseases. The model is stored in an AWS S3 bucket and the API is deployed on Render for testing purposes. 

## Supported Diseases

The model can predict the following crop diseases:

1. Anthracnose
2. Apple Scab
3. Black Spot
4. Blight
5. Blossom End Rot
6. Botrytis
7. Brown Rot
8. Canker
9. Cedar Apple Rust
10. Clubroot
11. Crown Gall
12. Downy Mildew
13. Fire Blight
14. Fusarium
15. Gray Mold
16. Leaf Spots
17. Mosaic Virus
18. Nematodes
19. Powdery Mildew
20. Verticillium

## Getting Started

### Prerequisites

- Python 3.10 
- TensorFlow
- FastAPI
- boto3 (for accessing AWS S3)
- requests (for making HTTP requests)

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Bhagyesh-20/api_ml.git
    cd api_ml
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Configuration

1. Set up your AWS credentials to access the S3 bucket where the model is stored. You can configure the AWS CLI with:
    ```sh
    aws configure
    ```
    Alternatively, set the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables.

2. Update the `config.py` file with your S3 bucket name and model filename:
    ```python
    S3_BUCKET = 'your-s3-bucket-name'
    MODEL_FILE = 'your-model-file.h5'
    ```

### Running the API

1. Start the Flask application:
    ```sh
    python app.py
    ```

2. The API will be available at `http://localhost:8000`.

### API Endpoints

#### Predict Disease

- **URL:** `/predict`
- **Method:** `POST`
- **Description:** Predicts the disease from an uploaded image.
- **Request:**
  - `file`: The image file of the crop leaf.
- **Response:**
  - `disease`: The predicted disease.

**Example:**

```sh
curl -X POST -F "file=@path/to/leaf.jpg" http://localhost:5000/predict
```

### Testing

You can use tools like Postman to test the API endpoints.

### Deployment

1. Ensure the necessary configurations for AWS and the model file are correctly set.
2. Deploy the application to Render or any other preferred platform. 

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.


---

For any issues or feature requests, please open an issue in the repository.
