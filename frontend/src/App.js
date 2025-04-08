import React, { useState } from 'react';
import axios from 'axios';
import { 
  Container, 
  Box, 
  Typography, 
  Button, 
  CircularProgress,
  Grid,
  Paper,
  FormControl,
  FormLabel,
  RadioGroup,
  Radio,
  FormControlLabel,
  TextField,
  Select,
  MenuItem,
  InputLabel
} from '@mui/material';
import './styles/styles.css';

function App() {
  // State management
  const [organType, setOrganType] = useState('brain');
  const [imageType, setImageType] = useState('mri');
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('No file selected');
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [chatMessage, setChatMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [error, setError] = useState('');

  // Handle file selection
  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setFileName(selectedFile.name);
    }
  };

  // Handle form submission
  const handleSubmit = async (event) => {
    event.preventDefault();
    
    if (!file) {
      setError('Please select an image file first');
      return;
    }
    
    setIsLoading(true);
    setError('');
    setChatHistory([]);
    
    // Create form data
    const formData = new FormData();
    formData.append('image', file);
    formData.append('organType', organType);
    formData.append('imageType', imageType);
    
    try {
      const response = await axios.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      setResults(response.data);
      
      // Add initial LLaMA response to chat history
      if (response.data.llamaResponse) {
        setChatHistory([
          { role: 'assistant', content: response.data.llamaResponse }
        ]);
      }
      
    } catch (error) {
      console.error('Error uploading image:', error);
      setError('Error processing image. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle sending chat message
  const handleSendMessage = async () => {
    if (!chatMessage.trim() || !results) return;
    
    // Add user message to chat history
    const newMessage = { role: 'user', content: chatMessage };
    setChatHistory([...chatHistory, newMessage]);
    setChatMessage('');
    
    try {
      const response = await axios.post('/api/chat', {
        message: chatMessage,
        prediction: results.prediction,
        organType: results.organType
      });
      
      // Add LLaMA response to chat history
      setChatHistory(prevHistory => [
        ...prevHistory,
        { role: 'assistant', content: response.data.response }
      ]);
      
    } catch (error) {
      console.error('Error sending message:', error);
      setChatHistory(prevHistory => [
        ...prevHistory,
        { role: 'assistant', content: 'Sorry, I encountered an error processing your message.' }
      ]);
    }
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }} className="header">
        <Typography variant="h2" component="h1" gutterBottom align="center">
          MedAI Vision
        </Typography>
      </Box>
      
      {/* Image Upload Form */}
      <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
        <form onSubmit={handleSubmit}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FormControl component="fieldset">
                <FormLabel component="legend">Select Organ Type</FormLabel>
                <RadioGroup
                  row
                  name="organ-type"
                  value={organType}
                  onChange={(e) => setOrganType(e.target.value)}
                >
                  <FormControlLabel value="brain" control={<Radio />} label="Brain" />
                  <FormControlLabel value="lungs" control={<Radio />} label="Lungs" />
                </RadioGroup>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel id="image-type-label">Image Type</InputLabel>
                <Select
                  labelId="image-type-label"
                  id="image-type"
                  value={imageType}
                  label="Image Type"
                  onChange={(e) => setImageType(e.target.value)}
                >
                  <MenuItem value="mri">MRI</MenuItem>
                  <MenuItem value="ct">CT</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12}>
              <Button
                variant="contained"
                component="label"
                fullWidth
              >
                Upload Image
                <input
                  type="file"
                  accept=".jpg,.jpeg,.png"
                  hidden
                  onChange={handleFileChange}
                />
              </Button>
              <Typography variant="body2" sx={{ mt: 1 }}>
                {fileName}
              </Typography>
            </Grid>
            
            <Grid item xs={12}>
              <Button
                type="submit"
                variant="contained"
                color="primary"
                fullWidth
                disabled={isLoading}
              >
                {isLoading ? <CircularProgress size={24} /> : 'Process Image'}
              </Button>
              {error && (
                <Typography color="error" sx={{ mt: 1 }}>
                  {error}
                </Typography>
              )}
            </Grid>
          </Grid>
        </form>
      </Paper>
      
      {/* Results Display */}
      {results && (
        <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
          <Typography variant="h5" gutterBottom>
            Analysis Results
          </Typography>
          
          <Typography variant="subtitle1" color="primary" gutterBottom>
            Prediction: {results.prediction}
          </Typography>
          
          <Grid container spacing={3} sx={{ mt: 2 }}>
            <Grid item xs={12} md={4}>
              <Typography variant="subtitle1" gutterBottom>Original Image</Typography>
              <img 
                src={results.originalImage} 
                alt="Original" 
                className="result-image"
              />
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Typography variant="subtitle1" gutterBottom>Enhanced Image (SRGAN)</Typography>
              <img 
                src={results.srImage} 
                alt="Super Resolution" 
                className="result-image"
              />
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Typography variant="subtitle1" gutterBottom>
                Translated Image ({imageType === 'mri' ? 'CT' : 'MRI'})
              </Typography>
              <img 
                src={results.cgImage} 
                alt="Translated" 
                className="result-image"
              />
            </Grid>
          </Grid>
        </Paper>
      )}
      
      {/* LLaMA Chat Interface */}
      {results && (
        <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
          <Typography variant="h5" gutterBottom>
            Medical Consultation
          </Typography>
          
          <Box className="chat-container">
            {chatHistory.map((message, index) => (
              <div 
                key={index} 
                className={`chat-message ${message.role === 'user' ? 'user-message' : 'assistant-message'}`}
              >
                <Typography variant="body1">
                  {message.content}
                </Typography>
              </div>
            ))}
          </Box>
          
          <Box sx={{ display: 'flex', mt: 2 }}>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="Ask a question about your results..."
              value={chatMessage}
              onChange={(e) => setChatMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
            />
            <Button
              variant="contained"
              onClick={handleSendMessage}
              sx={{ ml: 1 }}
            >
              Send
            </Button>
          </Box>
        </Paper>
      )}
    </Container>
  );
}

export default App;
