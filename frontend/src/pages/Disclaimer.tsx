import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';

const Disclaimer: React.FC = () => (
  <Card sx={{ p: 2, mt: 4 }}>
    <CardContent>
      <Typography variant="h5" gutterBottom>
        Disclaimer
      </Typography>
      <Box>
        <Typography variant="body2" paragraph>
          DermaAI is a research prototype and **does not provide medical diagnosis**. The predictions are for informational and educational purposes only and should not be used for clinical decision‑making. Always consult a qualified healthcare professional for medical advice.
        </Typography>
        <Typography variant="body2" paragraph>
          The developers and contributors are not liable for any decisions made based on the AI output.
        </Typography>
      </Box>
    </CardContent>
  </Card>
);

export default Disclaimer;
