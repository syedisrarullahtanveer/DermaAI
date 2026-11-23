import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';

const About: React.FC = () => (
  <Card sx={{ p: 2, mt: 4 }}>
    <CardContent>
      <Typography variant="h5" gutterBottom>
        About DermaAI
      </Typography>
      <Box>
        <Typography variant="body1" paragraph>
          DermaAI is an AI‑powered dermatology assistance tool that analyzes dermoscopic images to provide diagnostic suggestions and confidence scores. It combines state‑of‑the‑art deep learning models with explainable AI techniques such as Grad‑CAM to help clinicians and researchers understand model predictions.
        </Typography>
        <Typography variant="body2" paragraph>
          The project is open‑source and aims to accelerate research and development in computer‑aided dermatology.
        </Typography>
      </Box>
    </CardContent>
  </Card>
);

export default About;
