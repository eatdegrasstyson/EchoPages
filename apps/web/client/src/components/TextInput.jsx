const MAX_CHARS = 20_000;

export default function TextInput({ value, onChange, placeholder, rows = 12 }) {
  const count = value.length;
  const near = count > MAX_CHARS * 0.85;
  const over = count > MAX_CHARS;

  return (
    <div style={{ position: 'relative' }}>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        rows={rows}
        maxLength={MAX_CHARS}
      />
      <div style={{
        textAlign: 'right',
        fontSize: '0.75rem',
        marginTop: '0.25rem',
        color: over ? 'var(--color-tension, #e74c3c)' : near ? '#f39c12' : 'var(--text-secondary)',
      }}>
        {count.toLocaleString()} / {MAX_CHARS.toLocaleString()}
      </div>
    </div>
  );
}
