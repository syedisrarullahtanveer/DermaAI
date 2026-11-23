import React, { useState } from 'react';
import { Box, Button, Card, CardMedia, Typography } from '@mui/material';
import { Upload } from 'lucide-react';
import axios from 'axios';

// Adjust the base URL to match the FastAPI backend (assuming localhost:8000)
    const API_BASE = 'http://localhost:8000/api';

interface Props {
  onResult?: (data: any) => void; // callback to parent with prediction result
}

const UploadImage: React.FC<Props> = ({ onResult }) => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>('');
  const [loading, setLoading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) {
      setFile(selected);
      const url = URL.createObjectURL(selected);
      setPreview(url);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    const form = new FormData();
    form.append('file', file);
    try {
      const resp = await axios.post(`${API_BASE}/predict`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      onResult?.(resp.data);
    } catch (err) {
      console.error('Prediction error', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card sx={{ p: 2, mb: 3 }}>
      <Typography variant="h6" gutterBottom>
        Upload Dermoscopy Image
      </Typography>
      <Box display="flex" flexDirection="column" alignItems="center">
        <Button
          variant="outlined"
          component="label"
          startIcon={<Upload size={18} />}
          disabled={loading}
        >
          Choose File
          <input type="file" hidden accept="image/*" onChange={handleChange} />
        </Button>
        {preview && (
          <CardMedia
            component="img"
            src={preview}
            alt="preview"
            sx={{ maxHeight: 300, mt: 2, borderRadius: 1 }}
          />
        )}
        <Button
          variant="contained"
          color="primary"
          onClick={handleAnalyze}
          disabled={!file || loading}
          sx={{ mt: 2 }}
        >
          {loading ? 'Analyzing…' : 'Analyze'}
        </Button>
      </Box>
    </Card>
  );
};

export default UploadImage;
