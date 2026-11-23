import asyncio
import base64
import io
import os
from pathlib import Path
import pytest
import httpx
from PIL import Image

# Helper to create a simple red square PNG image in memory
def create_test_image() -> bytes:
    img = Image.new('RGB', (224, 224), color='red')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

@pytest.fixture(scope='session')
def anyio_backend():
    return 'asyncio'

@pytest.fixture(scope='session')
async def client():
    async with httpx.AsyncClient(base_url='http://localhost:8000') as c:
        yield c

@pytest.fixture(scope='session')
def test_image_path(tmp_path_factory):
    # Write the image to a temporary file
    img_bytes = create_test_image()
    path = tmp_path_factory.mktemp('data') / 'test.png'
    path.write_bytes(img_bytes)
    return path

@pytest.fixture(scope='session')
def invalid_file_path(tmp_path_factory):
    # Write a simple text file
    path = tmp_path_factory.mktemp('data') / 'invalid.txt'
    path.write_text('this is not an image')
    return path

@pytest.mark.anyio
async def test_successful_prediction(client, test_image_path):
    with open(test_image_path, 'rb') as f:
        files = {'file': ('test.png', f, 'image/png')}
        resp = await client.post('/api/predict', files=files)
    assert resp.status_code == 200
    json = resp.json()
    # Verify required fields
    assert 'top_prediction' in json
    assert 'predictions' in json
    assert isinstance(json['predictions'], list)
    # Grad-CAM should be present by default
    assert 'gradcam_base64' in json
    assert json['gradcam_base64'] is not None

@pytest.mark.anyio
async def test_invalid_image_handling(client, invalid_file_path):
    with open(invalid_file_path, 'rb') as f:
        files = {'file': ('invalid.txt', f, 'text/plain')}
        resp = await client.post('/api/predict', files=files)
    # Should return 400 Bad Request due to unsupported MIME type
    assert resp.status_code == 400
    json = resp.json()
    assert 'detail' in json

@pytest.mark.anyio
async def test_missing_file_field(client):
    # No file provided
    resp = await client.post('/api/predict')
    # FastAPI should report validation error (422)
    assert resp.status_code == 422

@pytest.mark.anyio
async def test_gradcam_disabled(client, test_image_path):
    with open(test_image_path, 'rb') as f:
        files = {'file': ('test.png', f, 'image/png')}
        params = {'include_gradcam': 'false'}
        resp = await client.post('/api/predict', params=params, files=files)
    assert resp.status_code == 200
    json = resp.json()
    # Grad-CAM disabled should return null
    assert json.get('gradcam_base64') is None
