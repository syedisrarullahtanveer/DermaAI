import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';

interface Props {
  gradcamBase64?: string;
}

const GradCamViewer: React.FC<Props> = ({ gradcamBase64 }) => {
  if (!gradcamBase64) {
    return <Typography variant="body2">Grad‑CAM not available.</Typography>;
  }

  return (
    <Card sx={{ p: 2, mb: 3 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Grad‑CAM Heatmap
        </Typography>
        <Box display="flex" justifyContent="center">
          <img
            src={`data:image/jpeg;base64,${gradcamBase64}`}
            alt="Grad‑CAM"
            style={{ maxWidth: '100%', maxHeight: 400, borderRadius: 4 }}
          />
        </Box>
      </CardContent>
    </Card>
  );
};

export default GradCamViewer;
