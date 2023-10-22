import json

import requests

BASE_URL = "http://10.42.101.106:1490"


def test_predict_endpoint():
    with open("../images/test.png", "rb") as image_file:
        files = {"image": image_file}
        response = requests.post(f"{BASE_URL}/predict", files=files)

    assert response.status_code == 200
    data = json.loads(response.content)
    print("Prediction:", data["prediction"])


def test_predict_image_path_endpoint():
    image_path = "../images/test.png"
    response = requests.get(f"{BASE_URL}/predict_image/{image_path}")

    assert response.status_code == 200
    data = json.loads(response.content)
    print("Prediction:", data["prediction"])


if __name__ == "__main__":
    test_predict_endpoint()
    test_predict_image_path_endpoint()
