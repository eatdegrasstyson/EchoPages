import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import TextInput from '../components/TextInput';

export default function UploadPage() {
  const [title, setTitle] = useState('');
  const [rawText, setRawText] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadMethod, setUploadMethod] = useState('text'); // 'text' or 'pdf'
  const navigate = useNavigate();

  function handleFileChange(e) {
    const file = e.target.files[0];
    if (file && file.type === 'application/pdf') {
      setSelectedFile(file);
    } else if (file) {
      alert('Please select a PDF file');
      e.target.value = '';
    }
  }

  function handleSubmit(e) {
    e.preventDefault();

    if (uploadMethod === 'text' && (!title.trim() || !rawText.trim())) {
      alert('Please provide both a title and text.');
      return;
    }

    if (uploadMethod === 'pdf' && !selectedFile) {
      alert('Please select a PDF file.');
      return;
    }

    // For demo purposes, just navigate to the sample reader
    // In a real implementation, this would process the text/PDF
    navigate('/read/1');
  }

  return (
    <div>
      <h2 style={{ marginBottom: '1.5rem' }}>Upload Text or PDF</h2>

      <div style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem' }}>
          <button
            type="button"
            className={uploadMethod === 'text' ? 'btn btn-primary' : 'btn'}
            onClick={() => setUploadMethod('text')}
          >
            Paste Text
          </button>
          <button
            type="button"
            className={uploadMethod === 'pdf' ? 'btn btn-primary' : 'btn'}
            onClick={() => setUploadMethod('pdf')}
          >
            Upload PDF
          </button>
        </div>
      </div>

      <form className="upload-form" onSubmit={handleSubmit}>
        {uploadMethod === 'text' ? (
          <>
            <div>
              <label htmlFor="title">Title</label>
              <input
                id="title"
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="My Story"
              />
            </div>

            <div>
              <label htmlFor="text">Text</label>
              <TextInput
                value={rawText}
                onChange={setRawText}
                placeholder="Paste your text here... Separate paragraphs with blank lines for best results."
              />
            </div>
          </>
        ) : (
          <div>
            <label htmlFor="pdf">Upload PDF</label>
            <input
              id="pdf"
              type="file"
              accept=".pdf"
              onChange={handleFileChange}
              style={{
                padding: '0.75rem',
                border: '2px dashed var(--border)',
                borderRadius: '8px',
                cursor: 'pointer',
                width: '100%',
              }}
            />
            {selectedFile && (
              <p style={{ marginTop: '0.5rem', color: 'var(--text-secondary)' }}>
                Selected: {selectedFile.name}
              </p>
            )}
          </div>
        )}

        <button type="submit" className="btn btn-primary">
          Continue to Reader
        </button>
      </form>
    </div>
  );
}
