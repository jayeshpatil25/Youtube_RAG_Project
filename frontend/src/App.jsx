import { useState, useRef, useEffect } from 'react'

const API_BASE = 'http://localhost:8000'

// ── Helpers ───────────────────────────────────────────────────────────────────
function formatTime(date) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

// ── Sub-components ────────────────────────────────────────────────────────────

function TypingIndicator() {
  return (
    <div className="message-row">
      <div className="avatar ai">🤖</div>
      <div className="typing-indicator">
        <div className="typing-dot" />
        <div className="typing-dot" />
        <div className="typing-dot" />
      </div>
    </div>
  )
}

function SourcesPanel({ sources }) {
  const [open, setOpen] = useState(false)
  if (!sources || sources.length === 0) return null
  return (
    <div>
      <button className="sources-toggle" onClick={() => setOpen(o => !o)}>
        📚 {open ? 'Hide' : 'Show'} {sources.length} sources {open ? '▲' : '▼'}
      </button>
      {open && (
        <div className="sources-list">
          {sources.map((src, i) => (
            <div className="source-item" key={i}>
              <div className="source-meta">
                <span className="source-rank">Chunk #{src.chunk_id}</span>
                <span className="source-score">RRF: {src.rrf_score}</span>
              </div>
              <span>{src.content}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function Message({ msg }) {
  const isUser = msg.role === 'user'
  return (
    <div className={`message-row ${isUser ? 'user' : ''}`}>
      <div className={`avatar ${isUser ? 'user' : 'ai'}`}>
        {isUser ? '👤' : '🤖'}
      </div>
      <div className="message-content">
        <div className={`message-bubble ${isUser ? 'user' : 'ai'}`}>
          {msg.content}
        </div>
        {msg.sources && <SourcesPanel sources={msg.sources} />}
        <span className="message-time">{formatTime(msg.time)}</span>
      </div>
    </div>
  )
}

function EmptyState({ onChipClick, videoReady }) {
  const suggestions = [
    'What is this video about?',
    'Summarize the key points',
    'What are the main takeaways?',
    'Explain the most important concept',
  ]
  return (
    <div className="empty-state">
      <div className="empty-orb">🎬</div>
      <h1 className="empty-title">VidChat AI</h1>
      <p className="empty-subtitle">
        {videoReady
          ? 'Your video is indexed! Ask anything about it below.'
          : 'Paste a YouTube URL in the sidebar to get started. The AI will semantically chunk and index it, then you can chat with it.'}
      </p>
      {videoReady && (
        <div className="suggestion-chips">
          {suggestions.map(s => (
            <button className="chip" key={s} onClick={() => onChipClick(s)}>{s}</button>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Sidebar ───────────────────────────────────────────────────────────────────
function Sidebar({ status, onLoad }) {
  const [url, setUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleLoad = async () => {
    if (!url.trim()) return
    setLoading(true)
    setError('')
    try {
      await onLoad(url.trim())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleKey = (e) => {
    if (e.key === 'Enter') handleLoad()
  }

  const isLoading = status.state === 'loading'
  const isReady = status.state === 'ready'
  const isError = status.state === 'error'

  return (
    <aside className="sidebar">
      <span className="sidebar-title">Video Source</span>

      {/* Video URL input */}
      <div className="video-card">
        <div className="video-card-header">
          <div className="video-card-icon">▶️</div>
          <div>
            <div className="video-card-title">YouTube Video</div>
            <div className="video-card-subtitle">Paste a video URL to index it</div>
          </div>
        </div>

        <div className="url-input-wrapper">
          <input
            id="youtube-url-input"
            className="url-input"
            type="url"
            placeholder="https://youtube.com/watch?v=..."
            value={url}
            onChange={e => setUrl(e.target.value)}
            onKeyDown={handleKey}
            disabled={isLoading}
          />
        </div>

        <button
          id="load-video-btn"
          className="load-btn"
          onClick={handleLoad}
          disabled={isLoading || !url.trim()}
        >
          {isLoading ? (
            <><span style={{ animation: 'spin 1s linear infinite', display: 'inline-block' }}>⏳</span> Processing...</>
          ) : (
            <><span>🚀</span> Load & Index Video</>
          )}
        </button>

        {error && (
          <div style={{ fontSize: '0.78rem', color: '#fca5a5', padding: '4px 2px' }}>
            ❌ {error}
          </div>
        )}
      </div>

      {/* Status */}
      {status.state !== 'idle' && (
        <div className="status-card">
          <div className="status-row">
            <span className="status-icon">
              {isReady ? '✅' : isError ? '❌' : '⚙️'}
            </span>
            <span className="status-text">{status.message}</span>
          </div>

          {isLoading && (
            <div className="status-bar-wrap">
              <div className="status-bar" style={{ width: '100%' }} />
            </div>
          )}

          {isReady && status.stats && (
            <div className="stats-grid">
              <div className="stat-item">
                <div className="stat-value">{status.stats.chunk_count ?? '—'}</div>
                <div className="stat-label">Chunks</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{status.stats.word_count?.toLocaleString() ?? '—'}</div>
                <div className="stat-label">Words</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{status.stats.avg_chunk_size ?? '—'}</div>
                <div className="stat-label">Avg Size</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{(status.stats.transcript_chars / 1000).toFixed(1)}k</div>
                <div className="stat-label">Characters</div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Tech stack info */}
      <span className="sidebar-title" style={{ marginTop: 'auto' }}>Powered By</span>
      <div className="tech-badges">
        <span className="tech-badge">🧠 Semantic Chunking</span>
        <span className="tech-badge cyan">🗄️ ChromaDB</span>
        <span className="tech-badge">🔍 BM25</span>
        <span className="tech-badge cyan">⚡ RRF Fusion</span>
        <span className="tech-badge">🤖 Groq LLaMA 3.1</span>
        <span className="tech-badge cyan">🤗 HuggingFace</span>
      </div>
    </aside>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [messages, setMessages] = useState([])
  const [inputText, setInputText] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [chatError, setChatError] = useState('')
  const [videoStatus, setVideoStatus] = useState({ state: 'idle', message: '', stats: {} })
  const [videoReady, setVideoReady] = useState(false)

  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)
  const pollRef = useRef(null)

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isTyping])

  // Poll status while loading
  const startPolling = () => {
    if (pollRef.current) clearInterval(pollRef.current)
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/status`)
        const data = await res.json()
        setVideoStatus(data)
        if (data.state === 'ready') {
          setVideoReady(true)
          clearInterval(pollRef.current)
        } else if (data.state === 'error') {
          clearInterval(pollRef.current)
        }
      } catch (_) {}
    }, 1500)
  }

  const handleLoadVideo = async (url) => {
    setVideoReady(false)
    setMessages([])
    setChatError('')
    setVideoStatus({ state: 'loading', message: 'Starting ingestion...', stats: {} })

    const res = await fetch(`${API_BASE}/api/load-video`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ youtube_url: url }),
    })
    if (!res.ok) {
      const err = await res.json()
      throw new Error(err.detail || 'Failed to start video ingestion')
    }
    startPolling()
  }

  const handleSend = async () => {
    const question = inputText.trim()
    if (!question || isTyping) return

    setChatError('')
    setInputText('')
    setMessages(prev => [...prev, { role: 'user', content: question, time: new Date() }])
    setIsTyping(true)

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Chat request failed')
      }
      const data = await res.json()
      setMessages(prev => [...prev, {
        role: 'ai',
        content: data.answer,
        sources: data.sources,
        time: new Date(),
      }])
    } catch (e) {
      setChatError(e.message)
    } finally {
      setIsTyping(false)
      inputRef.current?.focus()
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleChipClick = (text) => {
    setInputText(text)
    inputRef.current?.focus()
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-logo">
          <div className="logo-icon">🎬</div>
          <span className="logo-text">VidChat AI</span>
        </div>
        <div className="header-badge">
          <div className={`dot ${videoReady ? '' : 'inactive'}`} />
          {videoReady ? 'Video ready · Ask anything' : 'No video loaded'}
        </div>
      </header>

      <div className="main-layout">
        {/* Sidebar */}
        <Sidebar status={videoStatus} onLoad={handleLoadVideo} />

        {/* Chat Area */}
        <main className="chat-area">
          {messages.length === 0 && !isTyping ? (
            <EmptyState onChipClick={handleChipClick} videoReady={videoReady} />
          ) : (
            <div className="messages-container">
              {messages.map((msg, i) => <Message key={i} msg={msg} />)}
              {isTyping && <TypingIndicator />}
              <div ref={messagesEndRef} />
            </div>
          )}

          {chatError && (
            <div className="error-banner">
              ❌ {chatError}
            </div>
          )}

          {/* Input */}
          <div className="chat-input-area">
            <div className="input-wrapper">
              <textarea
                id="chat-input"
                ref={inputRef}
                className="chat-input"
                placeholder={videoReady ? 'Ask anything about the video...' : 'Load a video first to start chatting...'}
                value={inputText}
                onChange={e => setInputText(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={!videoReady || isTyping}
                rows={1}
              />
              <button
                id="send-btn"
                className="send-btn"
                onClick={handleSend}
                disabled={!videoReady || isTyping || !inputText.trim()}
                title="Send (Enter)"
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="22" y1="2" x2="11" y2="13" />
                  <polygon points="22 2 15 22 11 13 2 9 22 2" />
                </svg>
              </button>
            </div>
            <p className="input-hint">
              Press <kbd style={{ background: 'rgba(255,255,255,0.08)', padding: '1px 5px', borderRadius: '4px', fontSize: '0.68rem' }}>Enter</kbd> to send · <kbd style={{ background: 'rgba(255,255,255,0.08)', padding: '1px 5px', borderRadius: '4px', fontSize: '0.68rem' }}>Shift+Enter</kbd> for new line
            </p>
          </div>
        </main>
      </div>
    </div>
  )
}
