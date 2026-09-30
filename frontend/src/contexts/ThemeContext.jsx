import { createContext, useContext, useState, useEffect } from 'react'
const ThemeContext = createContext()

function initialTheme() {
  const saved = localStorage.getItem('theme')
  if (saved === 'light' || saved === 'dark') return saved
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(initialTheme)
  useEffect(() => {
    // 主题 class 挂在 <html>，与首屏防闪烁脚本保持一致；同时同步 body 以兼容旧选择器
    document.documentElement.className = theme
    document.body.className = theme
    localStorage.setItem('theme', theme)
    const meta = document.querySelector('meta[name="theme-color"]')
    if (meta) meta.setAttribute('content', theme === 'dark' ? '#05070f' : '#eef1f8')
  }, [theme])
  const toggleTheme = () => setTheme(t => t === 'light' ? 'dark' : 'light')
  return <ThemeContext.Provider value={{ theme, toggleTheme, setTheme }}>{children}</ThemeContext.Provider>
}
export function useTheme() { return useContext(ThemeContext) }
