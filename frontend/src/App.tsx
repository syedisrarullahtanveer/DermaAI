import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { Container, AppBar, Toolbar, Typography, Box, Button } from '@mui/material';
import UploadImage from './components/UploadImage';
import PredictionResult from './components/PredictionResult';
import GradCamViewer from './components/GradCamViewer';
import ModelInfo from './pages/ModelInfo';
import About from './pages/About';
import Disclaimer from './pages/Disclaimer';

const Home: React.FC = () => (
  <Box sx={{ mt: 4 }}>
    <UploadImage />
    <PredictionResult />
    <GradCamViewer />
  </Box>
);

const App: React.FC = () => {
  const [result, setResult] = React.useState<any>(null);

  const handleResult = (data: any) => {
    setResult(data);
  };

  const Home: React.FC = () => (
    <Box sx={{ mt: 4 }}>
      <UploadImage onResult={handleResult} />
      <PredictionResult result={result} />
      <GradCamViewer gradcamBase64={result?.gradcam} />
    </Box>
  );

  return (
    <Router>
      <AppBar position="static" color="primary">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            DermaAI Dashboard
          </Typography>
          <Button color="inherit" component={Link} to="/" style={{ textTransform: 'none' }}>
            Home
          </Button>
          <Button color="inherit" component={Link} to="/model" style={{ textTransform: 'none' }}>
            Model Info
          </Button>
          <Button color="inherit" component={Link} to="/about" style={{ textTransform: 'none' }}>
            About
          </Button>
          <Button color="inherit" component={Link} to="/disclaimer" style={{ textTransform: 'none' }}>
            Disclaimer
          </Button>
        </Toolbar>
      </AppBar>
      <Container maxWidth="lg">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/model" element={<ModelInfo />} />
          <Route path="/about" element={<About />} />
          <Route path="/disclaimer" element={<Disclaimer />} />
        </Routes>
      </Container>
    </Router>
  );
};
  return (
    <Router>
      <AppBar position="static" color="primary">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            DermaAI Dashboard
          </Typography>
          <Button color="inherit" component={Link} to="/" style={{ textTransform: 'none' }}>
            Home
          </Button>
          <Button color="inherit" component={Link} to="/model" style={{ textTransform: 'none' }}>
            Model Info
          </Button>
          <Button color="inherit" component={Link} to="/about" style={{ textTransform: 'none' }}>
            About
          </Button>
        </Toolbar>
      </AppBar>
      <Container maxWidth="lg">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/model" element={<ModelInfo />} />
          <Route path="/about" element={<About />} />
          <Route path="/disclaimer" element={<Disclaimer />} />
        </Routes>
      </Container>
    </Router>
  );
};

export default App;
