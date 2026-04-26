import './App.css'

function App() {
  const metrics = [
    { label: 'Policy modes', value: 'PPO + Heuristic + LLM' },
    { label: 'OpenEnv tasks', value: '3 benchmark tasks' },
    { label: 'Serving mode', value: 'HF Space Docker' },
    { label: 'Frontend', value: 'React + Vite dashboard' },
  ]

  return (
    <main className="page">
      <section className="hero">
        <p className="chip">TradeX / MEVerse</p>
        <h1>AMM Market Surveillance Dashboard</h1>
        <p className="subtitle">
          A lightweight React frontend for monitoring policy outcomes in the
          TradeX Hugging Face Space.
        </p>
        <div className="ctaRow">
          <a
            className="button primary"
            href="https://huggingface.co/spaces/Casp1an/TradeX"
            target="_blank"
            rel="noreferrer"
          >
            Open Hugging Face Space
          </a>
          <a className="button" href="https://huggingface.co/spaces/Casp1an/TradeX/tree/main" target="_blank" rel="noreferrer">
            View Space Files
          </a>
        </div>
      </section>

      <section className="grid">
        {metrics.map((item) => (
          <article key={item.label} className="card">
            <h2>{item.label}</h2>
            <p>{item.value}</p>
          </article>
        ))}
      </section>

      <section className="panel">
        <h2>Quick start</h2>
        <ol>
          <li>Build this frontend inside Docker.</li>
          <li>Serve static assets on port <code>7860</code>.</li>
          <li>Push to HF Space repo and deploy automatically.</li>
        </ol>
      </section>

      <section className="panel">
        <h2>Connected docs</h2>
        <div className="links">
          <a href="https://github.com/Frisbee148/TradeX-Hackathon/blob/main/README.md" target="_blank" rel="noreferrer">
            README story
          </a>
          <a href="https://github.com/Frisbee148/TradeX-Hackathon/blob/main/docs/hf-mini-blog.md" target="_blank" rel="noreferrer">
            HF mini blog
          </a>
          <a href="https://github.com/Frisbee148/TradeX-Hackathon/blob/main/docs/Dashboard.md" target="_blank" rel="noreferrer">
            Dashboard note
          </a>
        </div>
      </section>
    </main>
  )
}

export default App
