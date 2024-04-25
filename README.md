# Local Histogram Processing REST API

This is a simple Python REST API built with Flask that accepts an image as input, performs local histogram equalization on it and returns the image in the response

## Requirements

- Python 3.x
- Flask

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/annamal4/CS-712-final-project-backend.git
   ```

2. Navigate to the project directory

   ```
   cd CS-712-final-project-backend
   ```

3. Install dependencies using pip:

   ```
   pip3 install -r requirements.txt
   ```

## Usage

1. Start the flask server

    ```
    python3 app.py
    ```

2. The API will be accessible at http://localhost:5000/image.

3. Send a POST request to the endpoint http://localhost:5000/image with an image file attached.Example using CURL:

    ```
    curl -X POST -F "image=@test-img5.pgm" http://127.0.0.1:5000/image
    ```

    Replace test-img5.pgm with the actual path to the image file you want to process.

4. The API will perform local histogram equalization on the image and outputs it.

## Running using docker

1. Build image using docker

   ```
   docker build -t app .
   ```

2. Run image

   ```
   docker run -p 5000:5000 app
   ```