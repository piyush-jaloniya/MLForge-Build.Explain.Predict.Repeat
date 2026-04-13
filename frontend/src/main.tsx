import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'

const style = document.createElement('style')
style.textContent = `*,*::before,*::after{box-sizing:border-box}body,html{margin:0;padding:0}a{text-decoration:none}`
document.head.appendChild(style)

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
)
