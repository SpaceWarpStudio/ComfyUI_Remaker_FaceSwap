import requests
import time
from io import BytesIO
from PIL import Image, ImageOps, ImageSequence
import numpy as np
import torch
import threading

def pil2tensor(img):
    img = ImageOps.exif_transpose(img)
    image = img.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    return image

def load_image(image_source):
    if image_source.startswith('http'):
        response = requests.get(image_source)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(image_source)
    return img

def create_job(source_buffer, face_buffer, api_key):
    files = {
        'target_image': ('target_image.png', source_buffer, 'image/png'),
        'swap_image': ('swap_image.png', face_buffer, 'image/png')
    }

    response = requests.post(
        "https://developer.remaker.ai/api/remaker/v1/face-swap/create-job",
        headers={
            'accept': 'application/json',
            'Authorization': api_key,
        },
        files=files
    )

    response_json = response.json()
    if response.status_code != 200 or 'result' not in response_json or 'job_id' not in response_json['result']:
        raise Exception(f"API request failed with status code {response.status_code}: {response_json.get('message', {}).get('en', 'No error message provided')}")
    return response_json['result']['job_id']

def poll_for_result(job_id, api_key):
    fetch_job_url = f'https://developer.remaker.ai/api/remaker/v1/face-swap/{job_id}'
    while True:
        result_response = requests.get(fetch_job_url, headers={
            'accept': 'application/json',
            'Authorization': api_key,
        })
        result_json = result_response.json()
        if result_response.status_code == 200:
            if result_json['code'] == 100000 and 'output_image_url' in result_json['result']:
                return result_json['result']['output_image_url'][0]
            elif result_json['code'] == 300102:  # Image generation in progress
                time.sleep(5)  # Wait for 5 seconds before polling again
            elif result_json['code'] == 300104:  # Image generation failed
                raise Exception(f"Image generation failed with reason: {result_json['message']['en']}")
            else:
                raise Exception(f"Unexpected response code {result_json['code']}: {result_json['message']['en']}")
        else:
            raise Exception(f"Polling failed with status code {result_response.status_code}: {result_response.text}")

class RemakerFaceSwap:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image1": ("IMAGE",),
                "face_image1": ("IMAGE",),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "your_api_key_here"
                }),
            },
            "optional": {
                "source_image2": ("IMAGE",),
                "face_image2": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("Swapped Image 1", "Image URL 1", "Swapped Image 2", "Image URL 2")
    FUNCTION = "face_swap"
    CATEGORY = "FaceSwap"

    def face_swap(self, source_image1, face_image1, api_key, source_image2=None, face_image2=None):
        print(f"Input source_image1 type: {type(source_image1)}")
        print(f"Input source_image1 shape: {source_image1.shape if isinstance(source_image1, (np.ndarray, torch.Tensor)) else 'N/A'}")
        print(f"Input face_image1 type: {type(face_image1)}")
        print(f"Input face_image1 shape: {face_image1.shape if isinstance(face_image1, (np.ndarray, torch.Tensor)) else 'N/A'}")

        if source_image2 is not None and face_image2 is not None:
            print(f"Input source_image2 type: {type(source_image2)}")
            print(f"Input source_image2 shape: {source_image2.shape if isinstance(source_image2, (np.ndarray, torch.Tensor)) else 'N/A'}")
            print(f"Input face_image2 type: {type(face_image2)}")
            print(f"Input face_image2 shape: {face_image2.shape if isinstance(face_image2, (np.ndarray, torch.Tensor)) else 'N/A'}")

        def prepare_image(image):
            if isinstance(image, torch.Tensor):
                image_np = image.cpu().numpy()
            else:
                image_np = np.array(image)

            image_np = np.squeeze(image_np)

            if image_np.ndim == 2:
                image_np = np.stack([image_np] * 3, axis=-1)

            image_pil = Image.fromarray((image_np * 255).astype('uint8')).convert("RGB")
            buffer = BytesIO()
            image_pil.save(buffer, format="PNG")
            buffer.seek(0)
            return buffer

        source_buffer1 = prepare_image(source_image1)
        face_buffer1 = prepare_image(face_image1)
        job_id1 = create_job(source_buffer1, face_buffer1, api_key)

        if source_image2 is not None and face_image2 is not None:
            source_buffer2 = prepare_image(source_image2)
            face_buffer2 = prepare_image(face_image2)
            job_id2 = create_job(source_buffer2, face_buffer2, api_key)
        else:
            job_id2 = None

        url1, url2 = [None, None]

        def fetch_result(job_id, index):
            nonlocal url1, url2
            result_url = poll_for_result(job_id, api_key)
            if index == 1:
                url1 = result_url
            elif index == 2:
                url2 = result_url

        thread1 = threading.Thread(target=fetch_result, args=(job_id1, 1))
        thread1.start()
        if job_id2:
            thread2 = threading.Thread(target=fetch_result, args=(job_id2, 2))
            thread2.start()
            thread2.join()

        thread1.join()

        result_image1 = load_image(url1)
        result_image_tensor1 = pil2tensor(result_image1)

        if url2:
            result_image2 = load_image(url2)
            result_image_tensor2 = pil2tensor(result_image2)
        else:
            result_image_tensor2 = result_image_tensor1
            url2 = url1

        return (result_image_tensor1, url1, result_image_tensor2, url2)

NODE_CLASS_MAPPINGS = {
    "RemakerFaceSwap": RemakerFaceSwap
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemakerFaceSwap": "Remaker Face Swap"
}
