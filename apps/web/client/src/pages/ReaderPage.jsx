import { useState, useEffect, useMemo, useRef } from 'react';

const PAGE_CHAR_CAP = 1200;

function paginateSegments(segments, cap) {
  const pages = [];
  let current = [];
  let currentLen = 0;
  segments.forEach((seg, index) => {
    const len = seg.text.length;
    if (current.length > 0 && currentLen + len > cap) {
      pages.push(current);
      current = [];
      currentLen = 0;
    }
    current.push({ seg, index });
    currentLen += len;
  });
  if (current.length > 0) pages.push(current);
  return pages;
}
import { useParams, useLocation } from 'react-router-dom';
import { mockTexts } from '../data/mockData';
import EmotionTimeline from '../components/EmotionTimeline';
import SegmentCard from '../components/SegmentCard';
import AudioPlayer from '../components/AudioPlayer';
import { GEMS_COLORS } from '../utils/emotions';
import { useSpotify } from '../context/SpotifyContext';

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function top3Color(emotions) {
  const entries = Object.entries(emotions).sort((a, b) => b[1] - a[1]).slice(0, 3);
  let r = 0, g = 0, b = 0, total = 0;
  for (const [key, val] of entries) {
    const hex = GEMS_COLORS[key] || '#666666';
    const weight = val;
    total += weight;
    r += parseInt(hex.slice(1, 3), 16) * weight;
    g += parseInt(hex.slice(3, 5), 16) * weight;
    b += parseInt(hex.slice(5, 7), 16) * weight;
  }
  r = Math.round(r / total);
  g = Math.round(g / total);
  b = Math.round(b / total);
  return `rgb(${r}, ${g}, ${b})`;
}

function segmentBgColor(seg) {
  const entries = Object.entries(seg.emotions)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3);

  let r = 0, g = 0, b = 0, total = 0;
  for (const [key, val] of entries) {
    const hex = GEMS_COLORS[key] || '#666666';
    total += val;
    r += parseInt(hex.slice(1, 3), 16) * val;
    g += parseInt(hex.slice(3, 5), 16) * val;
    b += parseInt(hex.slice(5, 7), 16) * val;
  }

  r = Math.round(r / total);
  g = Math.round(g / total);
  b = Math.round(b / total);

  return `rgba(${r}, ${g}, ${b}, 0.25)`;
}

export default function ReaderPage() {
  const { id } = useParams();
  const location = useLocation();
  const [text, setText] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeIndex, setActiveIndex] = useState(0);
  const [pageIndex, setPageIndex] = useState(0);
  const [pageTurns, setPageTurns] = useState([]);
  const segmentRefs = useRef([]);
  const { isReady, playTrack } = useSpotify();

  const apiSegments = location.state?.segments || null;
  const pages = useMemo(
    () => (apiSegments ? paginateSegments(apiSegments, PAGE_CHAR_CAP) : []),
    [apiSegments]
  );

  useEffect(() => {
    if (apiSegments) {
      setLoading(false);
      return;
    }
    setTimeout(() => {
      setText(mockTexts[id] || mockTexts['1']);
      setLoading(false);
    }, 300);
  }, [id]);

  // Record the initial page-turn timestamp when the paged reader mounts
  useEffect(() => {
    if (apiSegments && pageTurns.length === 0) {
      setPageTurns([{ toPage: 0, at: Date.now() }]);
    }
  }, [apiSegments]); // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-play the matched song whenever the active segment changes
  useEffect(() => {
    if (!isReady || !apiSegments) return;
    const seg = apiSegments[activeIndex];
    if (!seg?.matchedSong?.spotifyID) return;
    playTrack(seg.matchedSong.spotifyID, seg.matchedSong.start ?? 0);
  }, [activeIndex, isReady]); // eslint-disable-line react-hooks/exhaustive-deps

  function goToPage(next) {
    if (next < 0 || next >= pages.length) return;
    setPageIndex(next);
    setPageTurns((prev) => [...prev, { toPage: next, at: Date.now() }]);
    const firstSegIdx = pages[next][0].index;
    setActiveIndex(firstSegIdx);
  }

  function jumpToSegment(segIdx) {
    const pIdx = pages.findIndex((page) => page.some((p) => p.index === segIdx));
    if (pIdx >= 0) goToPage(pIdx);
  }

  // Arrow-key navigation for the paged reader
  useEffect(() => {
    if (!apiSegments) return;
    function onKey(e) {
      if (e.key === 'ArrowRight') goToPage(pageIndex + 1);
      else if (e.key === 'ArrowLeft') goToPage(pageIndex - 1);
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [pageIndex, pages.length, apiSegments]); // eslint-disable-line react-hooks/exhaustive-deps

  function scrollToSegment(index) {
    setActiveIndex(index);
    segmentRefs.current[index]?.scrollIntoView({
      behavior: 'smooth',
      block: 'center',
    });
  }

  if (loading) return <div className="loading">Loading...</div>;

  // API flow: paged reader with inline colored spans
  if (apiSegments) {
    const currentPage = pages[pageIndex] || [];
    const atStart = pageIndex === 0;
    const atEnd = pageIndex >= pages.length - 1;

    return (
      <div>
        <h2 style={{ marginBottom: '0.5rem' }}>{location.state?.title || 'Your Text'}</h2>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
          {apiSegments.length} chunks &middot; {pages.length} pages
        </p>

        <EmotionTimeline
          segments={apiSegments}
          activeIndex={activeIndex}
          onSegmentClick={jumpToSegment}
        />

        <div
          className="card"
          style={{
            lineHeight: '2',
            fontSize: '1.05rem',
            marginTop: '1.5rem',
            minHeight: '60vh',
          }}
        >
          {currentPage.map(({ seg, index }) => {
            const bgColor = segmentBgColor(seg);

            return (
              <span
                key={index}
                style={{
                  backgroundColor: bgColor,
                  padding: '0.2em 0.4em',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  marginRight: '4px',
                  display: 'inline-block',
                }}
                onClick={() => setActiveIndex(index)}
                title={
                  `Top emotions: ${seg.dominant.join(', ')}\n` +
                  `Song: ${seg.matchedSong?.song_name || 'None'} ` +
                  `(${seg.matchedSong?.start_formatted || '--'} - ${seg.matchedSong?.end_formatted || '--'})\n\n` +
                  Object.entries(seg.emotions)
                    .map(([k, v]) => `${k}: ${v.toFixed(2)}`)
                    .join('\n')
                }
              >
                {seg.text}{' '}
              </span>
            );
          })}
        </div>

        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginTop: '1rem',
          }}
        >
          <button onClick={() => goToPage(pageIndex - 1)} disabled={atStart}>
            &larr; Prev
          </button>
          <span style={{ color: 'var(--text-secondary)' }}>
            Page {pageIndex + 1} of {pages.length}
          </span>
          <button onClick={() => goToPage(pageIndex + 1)} disabled={atEnd}>
            Next &rarr;
          </button>
        </div>

        <AudioPlayer />
      </div>
    );
  }

  // Mock data flow: existing behavior
  if (!text) return <div className="card">Text not found.</div>;

  const activeSong = text.segments[activeIndex]?.matchedSong || null;

  return (
    <div>
      <h2 style={{ marginBottom: '0.5rem' }}>{text.title}</h2>
      <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
        {text.segments.length} segments
      </p>

      <EmotionTimeline
        segments={text.segments}
        activeIndex={activeIndex}
        onSegmentClick={scrollToSegment}
      />

      <div>
        {text.segments.map((seg, i) => (
          <div key={i} ref={(el) => (segmentRefs.current[i] = el)}>
            <SegmentCard
              segment={seg}
              isActive={activeIndex === i}
              onClick={() => setActiveIndex(i)}
            />
          </div>
        ))}
      </div>

      <AudioPlayer />
    </div>
  );
}
