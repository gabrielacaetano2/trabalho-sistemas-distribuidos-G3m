import React, { useState, useEffect } from 'react'
import { 
  Search, 
  UploadCloud, 
  Palette, 
  TrendingUp, 
  Layers, 
  Database, 
  Activity, 
  User, 
  Calendar, 
  FileText, 
  Sparkles, 
  Image as ImageIcon,
  CheckCircle,
  HelpCircle,
  Clock,
  Trash2
} from 'lucide-react'

function App() {
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState('corporativo')
  const [isLoading, setIsLoading] = useState(false)
  const [loadingStep, setLoadingStep] = useState('')
  const [results, setResults] = useState([])
  const [explanation, setExplanation] = useState('')
  const [mcpInfo, setMcpInfo] = useState(null)
  const [isClearingCache, setIsClearingCache] = useState(false)
  
  // Imagem selecionada para detalhes na barra lateral
  const [selectedImage, setSelectedImage] = useState(null)
  
  // Todas as imagens cadastradas para exibição inicial
  const [allImages, setAllImages] = useState([])
  
  // Estado para o formulário de indexação
  const [indexFile, setIndexFile] = useState('')
  const [indexDesc, setIndexDesc] = useState('')
  const [indexAuthor, setIndexAuthor] = useState('')
  const [indexStatus, setIndexStatus] = useState(null)
  const [isIndexing, setIsIndexing] = useState(false)

  // Carregar imagens na inicialização
  useEffect(() => {
    fetchImages()
  }, [])

  const fetchImages = async () => {
    try {
      const res = await fetch('/api/images')
      if (res.ok) {
        const data = await res.json()
        setAllImages(data.images || [])
      }
    } catch (err) {
      console.error("Erro ao buscar imagens cadastradas:", err)
    }
  }

  const handleSearch = async (e) => {
    if (e) e.preventDefault()
    if (!query.trim()) return

    setIsLoading(true)
    setResults([])
    setExplanation('')
    setMcpInfo(null)
    setSelectedImage(null)

    // Simulando passos distribuídos para o usuário ver a orquestração
    const steps = [
      "Verificando cache de busca no Redis...",
      "Gerando embeddings textuais com OpenCLIP...",
      "Realizando busca vetorial de similaridade no Qdrant...",
      "Consultando metadados e regras de estilo no PostgreSQL...",
      "Invocando ferramentas analíticas no MCP Server...",
      "Gerando explicação RAG com Ollama (Llama 3)..."
    ]

    let stepIdx = 0
    setLoadingStep(steps[0])
    
    const stepInterval = setInterval(() => {
      if (stepIdx < steps.length - 1) {
        stepIdx++
        setLoadingStep(steps[stepIdx])
      }
    }, 900)

    try {
      const response = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, category })
      })

      clearInterval(stepInterval)

      if (response.ok) {
        const data = await response.json()
        setResults(data.results || [])
        setExplanation(data.explanation || '')
        setMcpInfo(data.mcp_info || null)
        
        // Seleciona automaticamente o primeiro resultado se houver
        if (data.results && data.results.length > 0) {
          setSelectedImage(data.results[0])
        }
      } else {
        alert("Erro na busca: " + response.statusText)
      }
    } catch (err) {
      clearInterval(stepInterval)
      console.error(err)
      alert("Erro ao conectar com o gateway do backend.")
    } finally {
      setIsLoading(false)
      setLoadingStep('')
    }
  }

  const handleIndex = async (e) => {
    e.preventDefault()
    if (!indexFile.trim() || !indexDesc.trim()) {
      setIndexStatus({ type: 'error', message: 'Preencha o nome do arquivo e a descrição.' })
      return
    }

    setIsIndexing(true)
    setIndexStatus(null)

    try {
      const response = await fetch('/api/index', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filename: indexFile,
          description: indexDesc,
          author: indexAuthor || 'Desconhecido'
        })
      })

      const data = await response.json()
      if (response.ok) {
        setIndexStatus({ type: 'success', message: `Indexado com sucesso! ID: ${data.id}` })
        setIndexFile('')
        setIndexDesc('')
        setIndexAuthor('')
        fetchImages() // Atualizar galeria inicial
      } else {
        setIndexStatus({ type: 'error', message: data.detail || 'Erro ao indexar.' })
      }
    } catch (err) {
      setIndexStatus({ type: 'error', message: 'Falha de comunicação com o backend.' })
    } finally {
      setIsIndexing(false)
    }
  }

  const handleClearCache = async () => {
    setIsClearingCache(true)

    try {
      const response = await fetch('/api/cache/clear', {
        method: 'POST'
      })

      const data = await response.json()

      if (response.ok) {
        alert(`Cache Redis limpo. Chaves removidas: ${data.deleted_keys}`)
      } else {
        alert(data.detail || 'Erro ao limpar cache.')
      }
    } catch (err) {
      console.error(err)
      alert('Falha de comunicação ao limpar cache.')
    } finally {
      setIsClearingCache(false)
    }
  }

  const handleQuickSearch = (text, cat) => {
    setQuery(text)
    if (cat) setCategory(cat)
    // Pequeno delay para atualizar os states antes de chamar a busca
    setTimeout(() => {
      document.getElementById('search-form-btn').click()
    }, 50)
  }

  return (
    <>
      <header>
        <div className="logo-container">
          <div className="logo-icon">G3</div>
          <div>
            <div className="logo-text">G3m</div>
            <div className="logo-sub">Distribuído & Semântico</div>
          </div>
        </div>
        <div className="header-actions">
          <button
            type="button"
            className="cache-clear-btn"
            onClick={handleClearCache}
            disabled={isClearingCache}
            title="Limpar cache de busca no Redis"
          >
            <Trash2 size={15} />
            {isClearingCache ? 'Limpando...' : 'Limpar cache'}
          </button>
          <div className="status-badge">
            <span className="status-dot"></span>
            Orquestrador Online
          </div>
        </div>
      </header>

      <main>
        {/* Hero Section */}
        <section className="hero">
          <h1>Motor de Busca de Imagens RAG</h1>
          <p>
            Encontre ativos visuais corporativos usando busca semântica multimodal (OpenCLIP), 
            validação de diretrizes de marca (RAG local no PostgreSQL/Ollama) e ferramentas externas de análise (MCP).
          </p>
        </section>

        {/* Search Bar Section */}
        <section className="search-container">
          <form onSubmit={handleSearch} className="search-box glass-panel">
            <div className="search-input-wrapper">
              <Search className="search-icon-left" size={22} />
              <input 
                type="text" 
                className="search-input" 
                placeholder="Ex: Equipe alegre cooperando no escritório moderno ou câmera clássica sobre fundo quente..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>
            <button type="submit" id="search-form-btn" className="search-button" disabled={isLoading}>
              <Sparkles size={18} />
              {isLoading ? "Processando..." : "Buscar"}
            </button>
          </form>

          {/* Categorias / Estilos de Marca */}
          <div className="category-filter">
            <button 
              type="button" 
              className={`category-btn ${category === 'corporativo' ? 'active' : ''}`}
              onClick={() => setCategory('corporativo')}
            >
              <Layers size={15} /> Estilo Corporativo
            </button>
            <button 
              type="button" 
              className={`category-btn ${category === 'tecnologia' ? 'active' : ''}`}
              onClick={() => setCategory('tecnologia')}
            >
              <Database size={15} /> Estilo Tecnologia
            </button>
            <button 
              type="button" 
              className={`category-btn ${category === 'retro' ? 'active' : ''}`}
              onClick={() => setCategory('retro')}
            >
              <Palette size={15} /> Estilo Retro
            </button>
            <button 
              type="button" 
              className={`category-btn ${category === 'minimalista' ? 'active' : ''}`}
              onClick={() => setCategory('minimalista')}
            >
              <TrendingUp size={15} /> Estilo Minimalista
            </button>
          </div>
        </section>

        {/* Loading Spinner with steps indicator */}
        {isLoading && (
          <div className="loading-container glass-panel">
            <div className="spinner"></div>
            <div className="loading-text">{loadingStep}</div>
            <div className="loading-subtext">Processando requisições em microsserviços paralelos</div>
          </div>
        )}

        {/* RAG Contextual Response Panel */}
        {!isLoading && explanation && (
          <div className="rag-explanation-container glass-panel">
            <div className="rag-header">
              <Sparkles className="rag-header-sparkle" size={18} />
              <span>Justificativa Inteligente (RAG - Llama 3)</span>
            </div>
            <div className="rag-text">"{explanation}"</div>
          </div>
        )}

        {/* Dynamic content grid (results + sidebar) */}
        {!isLoading && (results.length > 0 || allImages.length > 0) && (
          <div className="content-grid">
            <section className="gallery-section">
              <h2 className="section-title">
                {results.length > 0 ? "Resultados da Busca Semântica" : "Repositório de Imagens (Indexadas)"}
                <span className="results-count">
                  {results.length > 0 ? `${results.length} imagens encontradas` : `${allImages.length} imagens no total`}
                </span>
              </h2>

              <div className="image-grid">
                {(results.length > 0 ? results : allImages).map((img) => (
                  <div 
                    key={img.id} 
                    className={`image-card glass-panel ${selectedImage?.id === img.id ? 'selected' : ''}`}
                    onClick={() => setSelectedImage(img)}
                  >
                    <div className="card-img-wrapper">
                      <img src={img.url} alt={img.description} />
                      {img.score !== undefined && (
                        <div className="card-score">{(img.similarity).toFixed(1)}% similar</div>
                      )}
                    </div>
                    <div className="card-info">
                      <div className="card-title">{img.filename}</div>
                      <div className="card-desc">{img.description}</div>
                      <div className="card-footer">
                        <span style={{ display: 'flex', alignItems: 'center', gap: '3px' }}><User size={12} /> {img.author}</span>
                        {img.date_created && (
                          <span style={{ display: 'flex', alignItems: 'center', gap: '3px' }}><Clock size={12} /> {new Date(img.date_created).toLocaleDateString()}</span>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>

            {/* Sidebar for context detailed analysis */}
            <aside>
              {selectedImage ? (
                <div className="sidebar-panel glass-panel">
                  <div className="panel-title">
                    <Activity size={18} color="#a855f7" />
                    Análise do Ativo Visual
                  </div>
                  
                  <div className="info-item">
                    <span className="info-label">Nome do arquivo</span>
                    <span className="info-value" style={{ wordBreak: 'break-all', fontFamily: 'monospace' }}>
                      {selectedImage.filename}
                    </span>
                  </div>

                  <div className="info-item">
                    <span className="info-label">Autor da imagem</span>
                    <span className="info-value">{selectedImage.author}</span>
                  </div>

                  <div className="info-item">
                    <span className="info-label">Descrição Semântica (Embed)</span>
                    <span className="info-value">{selectedImage.description}</span>
                  </div>

                  {/* Cores dominantes trazidas via MCP tool */}
                  {results.length > 0 && mcpInfo?.color_palette && mcpInfo.color_palette.length > 0 && (
                    <div className="info-item">
                      <span className="info-label" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <Palette size={14} /> Paleta de Cores Dominantes (MCP)
                      </span>
                      <div className="colors-row">
                        {mcpInfo.color_palette.map((color, i) => (
                          <div key={i} className="color-swatch-wrapper">
                            <div className="color-swatch" style={{ backgroundColor: color }}></div>
                            <span className="color-hex">{color}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Tendências via Google Trends (MCP) */}
                  {results.length > 0 && mcpInfo?.google_trends?.top_queries && (
                    <div className="info-item">
                      <span className="info-label" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <TrendingUp size={14} /> Termos em Alta (Google Trends)
                      </span>
                      <div className="tag-cloud">
                        {mcpInfo.google_trends.top_queries.map((term, i) => (
                          <span key={i} className="tag">{term}</span>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="info-item" style={{ marginTop: '0.5rem', borderTop: '1px solid var(--panel-border)', paddingTop: '1rem' }}>
                    <span className="info-label">Caminho no Docker</span>
                    <span className="info-value" style={{ fontSize: '0.8rem', fontFamily: 'monospace', color: 'var(--text-muted)' }}>
                      {selectedImage.path}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="sidebar-panel glass-panel" style={{ alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', padding: '3rem 1.5rem', textAlign: 'center' }}>
                  <HelpCircle size={32} style={{ color: 'var(--text-muted)', marginBottom: '0.5rem' }} />
                  <p>Selecione uma imagem da galeria para inspecionar os metadados e análise do MCP.</p>
                </div>
              )}
            </aside>
          </div>
        )}

        {/* Dashboard de Indexação de Ativos */}
        <section className="glass-panel" style={{ maxWidth: '800px', margin: '2rem auto 0 auto', width: '100%' }}>
          <div className="panel-title" style={{ borderBottom: '1px solid var(--panel-border)', padding: '1.25rem 1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <UploadCloud size={20} color="#10b981" />
            Indexador de Novos Ativos de Imagem
          </div>
          
          <form onSubmit={handleIndex} className="upload-panel">
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '-0.5rem' }}>
              Para simular a indexação em Sistemas Distribuídos: Copie o arquivo da imagem para a pasta física 
              <code>data/images/</code> no seu computador e preencha os dados abaixo para gerar os embeddings OpenCLIP e salvar os dados.
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div className="form-group">
                <label>Nome exato do arquivo</label>
                <input 
                  type="text" 
                  placeholder="Ex: foto_reuniao.jpg" 
                  value={indexFile}
                  onChange={(e) => setIndexFile(e.target.value)}
                  required
                />
              </div>
              <div className="form-group">
                <label>Autor / Fotógrafo</label>
                <input 
                  type="text" 
                  placeholder="Ex: Gabriela Caetano" 
                  value={indexAuthor}
                  onChange={(e) => setIndexAuthor(e.target.value)}
                />
              </div>
            </div>

            <div className="form-group">
              <label>Descrição Semântica Completa (Para indexação do modelo CLIP)</label>
              <textarea 
                rows="3" 
                placeholder="Descreva em detalhes o que está na imagem para que a IA de busca possa encontrá-la (ex: Grupo de desenvolvedores bebendo café no intervalo em volta de um sofá azul no saguão...)"
                value={indexDesc}
                onChange={(e) => setIndexDesc(e.target.value)}
                required
              ></textarea>
            </div>

            <button type="submit" className="submit-btn" disabled={isIndexing}>
              <CheckCircle size={18} />
              {isIndexing ? "Extraindo Embeddings e Indexando..." : "Indexar Ativo Visual"}
            </button>

            {indexStatus && (
              <div style={{ 
                padding: '0.75rem 1rem', 
                borderRadius: '8px', 
                fontSize: '0.9rem',
                backgroundColor: indexStatus.type === 'success' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                border: `1px solid ${indexStatus.type === 'success' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)'}`,
                color: indexStatus.type === 'success' ? '#10b981' : '#ef4444',
                textAlign: 'center'
              }}>
                {indexStatus.message}
              </div>
            )}
          </form>
        </section>

        {/* Quick Search helpers for demonstration */}
        <section className="glass-panel" style={{ maxWidth: '800px', margin: '1rem auto 0 auto', width: '100%', padding: '1.25rem 1.5rem' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: '700', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
            💡 Sugestões de Busca RAG para Apresentação
          </h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            <button 
              onClick={() => handleQuickSearch("equipe reunida no escritório", "corporativo")}
              className="category-btn" 
              style={{ fontSize: '0.8rem', padding: '0.35rem 0.75rem' }}
            >
              "Equipe reunida" (Corporativo)
            </button>
            <button 
              onClick={() => handleQuickSearch("telas com linhas de código de computação", "tecnologia")}
              className="category-btn" 
              style={{ fontSize: '0.8rem', padding: '0.35rem 0.75rem' }}
            >
              "Código e telas" (Tecnologia)
            </button>
            <button 
              onClick={() => handleQuickSearch("camera antiga vintage analógica", "retro")}
              className="category-btn" 
              style={{ fontSize: '0.8rem', padding: '0.35rem 0.75rem' }}
            >
              "Câmera clássica" (Retro)
            </button>
            <button 
              onClick={() => handleQuickSearch("computador em mesa limpa com café", "minimalista")}
              className="category-btn" 
              style={{ fontSize: '0.8rem', padding: '0.35rem 0.75rem' }}
            >
              "Ambiente minimalista" (Minimalista)
            </button>
          </div>
        </section>
      </main>

      <footer style={{ borderTop: '1px solid var(--panel-border)', padding: '2rem 1.5rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.85rem', background: 'rgba(11, 15, 25, 0.4)' }}>
        G3m Smart Search Engine &copy; 2026. Trabalho de Sistemas Distribuídos (Fase 2 - Protótipo RAG/MCP).
      </footer>
    </>
  )
}

export default App
