import React from 'react';
import { Box, Card, CardContent, Typography, LinearProgress } from '@mui/material';

interface Props {
  result?: {
    label: string;
    confidence: number;
    probabilities: Record<string, number>;
    gradcam?: string; // base64 image
  };
}

const PredictionResult: React.FC<Props> = ({ result }) => {
  if (!result) {
    return <Typography variant="body1">No prediction yet.</Typography>;
  }

  const { label, confidence, probabilities } = result;

  return (
    <Card sx={{ p: 2, mb: 3 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Prediction Result
        </Typography>
        <Box display="flex" alignItems="center" mb={1}>
          <Typography variant="subtitle1" sx={{ mr: 1 }}>
            {label}
          </Typography>
          <LinearProgress
            variant="determinate"
            value={confidence * 100}
            sx={{ flexGrow: 1, height: 10, borderRadius: 5 }}
          />
          <Typography variant="body2" sx={{ ml: 1 }}>
            {(confidence * 100).toFixed(1)}%
          </Typography>
        </Box>
        <Box mt={2}>
          <Typography variant="subtitle2">Class probabilities:</Typography>
          {Object.entries(probabilities).map(([cls, prob]) => (
            <Box key={cls} display="flex" alignItems="center" mb={0.5}>
              <Typography variant="body2" sx={{ width: 120 }}>
                {cls}
              </Typography>
              <LinearProgress
                variant="determinate"
                value={prob * 100}
                sx={{ flexGrow: 1, height: 8, borderRadius: 4, mr: 1 }}
              />
              <Typography variant="caption">{(prob * 100).toFixed(1)}%</Typography>
            </Box>
          ))}
        </Box>
      </CardContent>
    </Card>
  );
};

export default PredictionResult;
