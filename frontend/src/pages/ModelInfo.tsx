import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';

const ModelInfo: React.FC = () => (
  <Card sx={{ p: 2, mt: 4 }}>
    <CardContent>
      <Typography variant="h5" gutterBottom>
        Model Information
      </Typography>
      <Box>
        <Typography variant="body1" paragraph>
          DermaAI uses a hybrid attention‑augmented architecture based on EfficientNet‑B0 as the backbone, combined with channel‑ and spatial‑attention (CBAM) modules and a multimodal metadata fusion network. The model was trained on a curated dataset of dermoscopic images covering seven diagnostic classes.
        </Typography>
        <Typography variant="body2" paragraph>
          <strong>Key specifications:</strong>
        </Typography>
        <ul>
          <li>Backbone: EfficientNet‑B0 (pre‑trained on ImageNet)</li>
          <li>Input size: 224×224 RGB</li>
          <li>Classes: 7 (AKIEC, BCC, BKL, DF, MEL, NV, VASC)</li>
          <li>Attention: CBAM (Channel & Spatial) + SE blocks</li>
          <li>Training epochs: 50 with cosine‑annealing LR schedule</li>
          <li>Grad‑CAM for explainability</li>
        </ul>
      </Box>
    </CardContent>
  </Card>
);

export default ModelInfo;
