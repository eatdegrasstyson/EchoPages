import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import TextInput from '../components/TextInput';

export default function UploadPage() {
  const [title, setTitle] = useState('');
  const [rawText, setRawText] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadMethod, setUploadMethod] = useState('text'); // 'text' or 'pdf'
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
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

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);

    if (uploadMethod === 'text') {
      if (!title.trim() || !rawText.trim()) {
        alert('Please provide both a title and text.');
        return;
      }

      setLoading(true);
      try {
        const res = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: rawText }),
        });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.error || `Server error ${res.status}`);
        }
        const { segments } = await res.json();
        navigate('/read/result', { state: { segments, title: title.trim() } });
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
      return;
    }

    if (uploadMethod === 'pdf' && !selectedFile) {
      alert('Please select a PDF file.');
      return;
    }

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

        {error && (
          <p style={{ color: 'var(--color-tension, #e74c3c)', marginTop: '0.5rem' }}>
            Error: {error}
          </p>
        )}

        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? 'Analyzing...' : 'Continue to Reader'}
        </button>
      </form>
    </div>
  );
}
