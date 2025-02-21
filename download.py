import os
import requests
from tqdm import tqdm

# Ensure the 'weights' directory exists
os.makedirs("weights", exist_ok=True)

# URL of the weight file
url = "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt"
weights_path = "weights/yolov9-c.pt"

# Downloading the weights with a progress bar
response = requests.get(url, stream=True)
total_size = int(response.headers.get('content-length', 0))
block_size = 1024  # 1KB
t = tqdm(total=total_size, unit='iB', unit_scale=True)

with open(weights_path, 'wb') as f:
    for data in response.iter_content(block_size):
        t.update(len(data))
        f.write(data)

t.close()
print("Download complete!")
